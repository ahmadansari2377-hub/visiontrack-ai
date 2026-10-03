"""
VisionTrack AI v2 - web server.

Run:  python app.py            (or double-click run.bat on Windows)
Open: http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
os.chdir(BASE)  # model downloads and relative paths always land in the project folder
CONFIG_PATH = BASE / "config.json"
VIDEOS_DIR = BASE / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)
(BASE / "logs").mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "source": "videos/traffic.mp4",
    "mode": "fast",
    "confidence": 0.3,
    "line": [0.05, 0.6, 0.95, 0.6],
    "loop": True,
    "use_openvino": True,
    "host": "127.0.0.1",
    "port": 5000,
}
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

# ---------------------------------------------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(BASE / "logs" / "visiontrack.log", maxBytes=2_000_000,
                            backupCount=3, encoding="utf-8"),
    ],
)
log = logging.getLogger("visiontrack")
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ---------------------------------------------------------------- config
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("config.json is broken (%s) - using defaults", e)
    # first run: pick any video that is already in the videos folder
    src = Path(cfg["source"])
    if not str(cfg["source"]).isdigit() and "://" not in str(cfg["source"]):
        if not (src if src.is_absolute() else BASE / src).exists():
            vids = sorted(p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXT)
            if vids:
                cfg["source"] = f"videos/{vids[0].name}"
    return cfg


CONFIG = load_config()
_cfg_lock = threading.Lock()


def save_config():
    with _cfg_lock:
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
        tmp.replace(CONFIG_PATH)  # atomic: config can't be half-written


from engine import MODES, VisionEngine  # noqa: E402  (after logging setup)

engine = VisionEngine(BASE, CONFIG, save_config)

# ---------------------------------------------------------------- flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3  # 2 GB uploads


def ok(**extra):
    return jsonify({"ok": True, **extra})


def fail(msg, code=400):
    return jsonify({"ok": False, "error": str(msg)}), code


def body() -> dict:
    """JSON request body, always a dict (bad or missing JSON -> {})."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.errorhandler(Exception)
def unexpected_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return fail(e.description, e.code)
    log.exception("Unexpected error in %s", request.path)
    return fail(f"Unexpected error: {e}", 500)


@app.route("/")
def index():
    videos = sorted(p.name for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXT)
    return render_template("index.html", modes=MODES, videos=videos)


@app.route("/video_feed")
def video_feed():
    def gen():
        seq = -1
        while True:
            jpeg, new_seq = engine.wait_jpeg(seq, timeout=1.0)
            if jpeg is None or new_seq == seq:
                continue
            seq = new_seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})


@app.route("/snapshot.jpg")
def snapshot():
    jpeg, _ = engine.wait_jpeg(-1, timeout=0)
    if jpeg is None:
        return fail("No frame yet", 503)
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/stats")
def api_stats():
    return jsonify(engine.stats())


@app.route("/api/health")
def api_health():
    s = engine.stats()
    return jsonify({"status": s["status"], "error": s["error"], "fps": s["processing_fps"]})


@app.route("/api/mode", methods=["POST"])
def api_mode():
    try:
        engine.set_mode(body().get("mode", ""))
        return ok()
    except ValueError as e:
        return fail(e)


@app.route("/api/confidence", methods=["POST"])
def api_confidence():
    try:
        engine.set_confidence(body().get("confidence"))
        return ok()
    except (TypeError, ValueError) as e:
        return fail(e)


@app.route("/api/line", methods=["POST"])
def api_line():
    try:
        engine.set_line(body().get("line", []))
        return ok()
    except (TypeError, ValueError) as e:
        return fail(e)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.reset_counts()
    return ok()


@app.route("/api/pause", methods=["POST"])
def api_pause():
    engine.set_paused(bool(body().get("paused")))
    return ok(paused=engine.paused)


@app.route("/api/source", methods=["POST"])
def api_source():
    """Switch to an existing video, a webcam number ('0') or a stream URL."""
    src = str(body().get("source", "")).strip()
    if not src:
        return fail("Source is empty")
    if not src.isdigit() and "://" not in src:
        name = secure_filename(Path(src).name)
        if not (VIDEOS_DIR / name).exists():
            return fail(f"Video '{name}' not found")
        src = f"videos/{name}"
    engine.set_source(src)
    return ok(source=src)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return fail("No file received")
    name = secure_filename(f.filename)
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        return fail("Unsupported file type. Use MP4, AVI, MOV, MKV or WEBM.")
    dest = VIDEOS_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    f.save(tmp)
    tmp.replace(dest)
    engine.set_source(f"videos/{name}")
    log.info("Uploaded %s (%.1f MB)", name, dest.stat().st_size / 1e6)
    return ok(source=f"videos/{name}")


@app.route("/api/export.csv")
def api_export():
    p = engine.export_csv_path()
    if not p.exists():
        return fail("No vehicles counted yet", 404)
    return send_file(p, as_attachment=True, download_name="visiontrack_crossings.csv")


@app.errorhandler(413)
def too_large(_):
    return fail("File is too large (max 2 GB)", 413)


# ---------------------------------------------------------------- main
def main():
    host, port = CONFIG.get("host", "127.0.0.1"), int(CONFIG.get("port", 5000))
    engine.start()

    def shutdown(*_):
        log.info("Shutting down...")
        engine.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("VisionTrack AI running at http://%s:%d", host, port)
    try:
        from waitress import serve  # production-grade server
        serve(app, host=host, port=port, threads=16, channel_timeout=30)
    except ImportError:
        log.warning("waitress not installed - using Flask's development server")
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    except OSError as e:
        log.error("Could not start on port %d: %s. Is VisionTrack already running?", port, e)
        engine.stop()
        time.sleep(0.5)
        sys.exit(1)


if __name__ == "__main__":
    main()
