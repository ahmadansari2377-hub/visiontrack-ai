"""
VisionTrack AI v2 - detection, tracking and counting engine.

Design goals
------------
* Boxes are always drawn on the exact frame they were detected on (no lag).
* Video plays in real time: if the CPU is slower than the video, frames are
  skipped instead of the video turning into slow motion.
* One vehicle = one count: a car is counted once, only when its centre really
  crosses the counting line, and only after it has been tracked for a few frames.
* Nothing crashes the server: bad frames, missing files, broken cameras and
  model problems are logged and retried.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("visiontrack.engine")

# COCO class ids that are vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {  # BGR
    "car": (255, 170, 40),
    "motorcycle": (60, 200, 255),
    "bus": (80, 80, 255),
    "truck": (200, 90, 220),
}

# Speed / accuracy presets. imgsz must be a multiple of 32.
MODES = {
    "fast": {"weights": "yolo11n.pt", "imgsz": 640, "label": "Fast (YOLO11n @640)"},
    "balanced": {"weights": "yolo11s.pt", "imgsz": 640, "label": "Balanced (YOLO11s @640)"},
    "accurate": {"weights": "yolo11m.pt", "imgsz": 800, "label": "Accurate (YOLO11m @800)"},
}

MAX_PROCESS_WIDTH = 1280  # frames are resized to this width before anything else
MIN_HITS_TO_COUNT = 3  # a track must be seen this many times before it can be counted


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
def parse_source(src: str):
    """'0' -> webcam 0, URLs stay strings, everything else is a file path."""
    s = str(src).strip()
    if s.isdigit():
        return int(s), True
    if s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return s, True
    return s, False


class FileSource:
    """Plays a video file in real time, skipping frames when the CPU falls behind."""

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else 30.0
        self.index = 0  # index of the next frame to be read
        self.skipped = 0
        self.looped = False
        self._restart_clock()

    def _restart_clock(self):
        self.t0 = time.perf_counter()
        self.i0 = self.index

    def resume(self):
        self._restart_clock()

    def read(self):
        """Return (frame, video_time_s) or (None, None) on end/failure."""
        self.looped = False
        target = self.i0 + int((time.perf_counter() - self.t0) * self.fps)
        # Behind real time -> skip frames cheaply with grab()
        skipped_now = 0
        while self.index < target and skipped_now < int(self.fps * 2):
            if not self.cap.grab():
                break
            self.index += 1
            skipped_now += 1
        self.skipped += skipped_now

        ok, frame = self.cap.read()
        if not ok or frame is None:
            if not self.loop:
                return None, None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0
            self._restart_clock()
            self.looped = True
            ok, frame = self.cap.read()
            if not ok:
                return None, None
        t = self.index / self.fps
        self.index += 1

        # Ahead of real time -> wait so the video doesn't play too fast
        due = self.t0 + (self.index - self.i0) / self.fps
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, 0.5))
        return frame, t

    def release(self):
        self.cap.release()


class LiveSource:
    """Webcam / RTSP / HTTP stream. A reader thread keeps only the newest frame,
    so we never process stale buffered frames and latency stays low."""

    def __init__(self, src):
        self.src = src
        backend = cv2.CAP_DSHOW if (os.name == "nt" and isinstance(src, int)) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera/stream: {src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else 30.0
        self.skipped = 0
        self.looped = False
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._failed = False
        self._lock = threading.Lock()
        self._running = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    self._failed = True
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def resume(self):
        pass

    def read(self):
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            if self._failed:
                return None, None
            with self._lock:
                if self._seq != self._last_seq and self._frame is not None:
                    self.skipped += max(0, self._seq - self._last_seq - 1)
                    self._last_seq = self._seq
                    return self._frame.copy(), time.perf_counter() - self._t0
            time.sleep(0.003)
        return None, None

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------- #
# Track bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    votes: Counter = field(default_factory=Counter)
    hits: int = 0
    last_seen: int = 0
    side: int = 0
    counted: bool = False
    box: np.ndarray | None = None  # smoothed box for drawing
    trail: deque = field(default_factory=lambda: deque(maxlen=25))
    flash: int = 0  # frames left to highlight after being counted

    @property
    def label(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "car"


def _side_of_line(p, a, b) -> int:
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _segment_intersect(p1, p2, a, b) -> bool:
    """True if the path p1->p2 crosses the line segment a->b."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = orient(a, b, p1), orient(a, b, p2)
    d3, d4 = orient(p1, p2, a), orient(p1, p2, b)
    return (d1 * d2 < 0) and (d3 * d4 <= 0)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class VisionEngine:
    def __init__(self, base_dir: Path, config: dict, save_config):
        self.base = Path(base_dir)
        self.cfg = config
        self._save_config = save_config
        self.models_dir = self.base / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.base / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.tracker_cfg = str(self.base / "tracker.yaml")

        self._lock = threading.RLock()
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_seq = 0

        self._running = False
        self._thread: threading.Thread | None = None
        self._reload_model = True
        self._reload_source = True
        self.paused = False

        self.model = None
        self.backend = "-"
        self.status = "starting"
        self.error = ""

        self.reset_counts()
        self._publish(self._message_frame("Starting VisionTrack AI..."))

    # ---------------- public API (called from Flask threads) -------------- #
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def reset_counts(self):
        with self._lock:
            self.counts = {"in": Counter(), "out": Counter()}
            self.events: deque = deque(maxlen=200)
            self.tracks: dict[int, TrackState] = {}
            self.unique_ids: set[int] = set()

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'")
        with self._lock:
            self.cfg["mode"] = mode
            self._reload_model = True
        self._save_config()

    def set_source(self, source: str):
        with self._lock:
            self.cfg["source"] = str(source)
            self._reload_source = True
        self.reset_counts()
        self._save_config()

    def set_line(self, pts):
        if not isinstance(pts, (list, tuple)) or len(pts) != 4:
            raise ValueError("Line needs 4 numbers: [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [float(v) for v in pts]
        for v in (x1, y1, x2, y2):
            if not 0.0 <= v <= 1.0:
                raise ValueError("Line coordinates must be between 0 and 1")
        if abs(x1 - x2) + abs(y1 - y2) < 0.02:
            raise ValueError("Line is too short")
        with self._lock:
            self.cfg["line"] = [x1, y1, x2, y2]
            for st in self.tracks.values():
                st.side = 0
        self._save_config()

    def set_confidence(self, conf: float):
        conf = float(conf)
        if not 0.05 <= conf <= 0.9:
            raise ValueError("Confidence must be between 0.05 and 0.9")
        with self._lock:
            self.cfg["confidence"] = conf
        self._save_config()

    def set_paused(self, paused: bool):
        self.paused = bool(paused)

    def wait_jpeg(self, last_seq: int, timeout: float = 1.0):
        with self._cond:
            if self._frame_seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._frame_seq

    def stats(self) -> dict:
        with self._lock:
            by_class = {}
            for name in VEHICLE_CLASSES.values():
                i, o = self.counts["in"][name], self.counts["out"][name]
                by_class[name] = {"in": i, "out": o, "total": i + o}
            total_in = sum(self.counts["in"].values())
            total_out = sum(self.counts["out"].values())
            src = self._source_obj if hasattr(self, "_source_obj") else None
            return {
                "status": self.status,
                "error": self.error,
                "paused": self.paused,
                "mode": self.cfg.get("mode"),
                "mode_label": MODES.get(self.cfg.get("mode"), {}).get("label", ""),
                "backend": self.backend,
                "source": self.cfg.get("source"),
                "confidence": self.cfg.get("confidence"),
                "line": self.cfg.get("line"),
                "counts": by_class,
                "total_in": total_in,
                "total_out": total_out,
                "total": total_in + total_out,
                "unique_vehicles": len(self.unique_ids),
                "active_tracks": self._active_tracks,
                "processing_fps": round(self._proc_fps, 1),
                "video_fps": round(src.fps, 1) if src else 0,
                "frames_analysed_pct": round(self._analysed_pct, 1),
                "inference_ms": round(self._infer_ms, 1),
                "events": list(self.events)[-15:][::-1],
            }

    def export_csv_path(self) -> Path:
        return self.logs_dir / "crossings.csv"

    # ---------------- model loading ---------------------------------------- #
    def _load_model(self):
        from ultralytics import YOLO

        mode = self.cfg.get("mode", "fast")
        preset = MODES.get(mode, MODES["fast"])
        weights = self.models_dir / preset["weights"]
        imgsz = preset["imgsz"]
        self._set_status("loading", f"Loading {preset['label']} (first run downloads the model)...")

        if not weights.exists():
            # Ultralytics downloads into the current directory; move it into models/
            m = YOLO(preset["weights"])
            src = Path(preset["weights"])
            if src.exists():
                shutil.move(str(src), str(weights))
            del m

        model, backend = None, "PyTorch CPU"
        if self.cfg.get("use_openvino", True):
            ov_dir = self.models_dir / f"{weights.stem}_{imgsz}_openvino_model"
            try:
                if not (ov_dir / f"{weights.stem}.xml").exists() and not any(ov_dir.glob("*.xml")):
                    self._publish(self._message_frame(
                        "Optimising model for your Intel CPU (one time, ~1 min)..."))
                    exported = YOLO(str(weights)).export(format="openvino", imgsz=imgsz,
                                                          half=False, dynamic=False, verbose=False)
                    if ov_dir.exists():
                        shutil.rmtree(ov_dir)
                    shutil.move(str(exported), str(ov_dir))
                model = YOLO(str(ov_dir), task="detect")
                backend = "OpenVINO (Intel optimised)"
            except Exception as e:  # noqa: BLE001
                log.warning("OpenVINO unavailable, using PyTorch instead: %s", e)
                model = None

        if model is None:
            try:
                import torch
                torch.set_num_threads(max(1, os.cpu_count() or 4))
            except Exception:  # noqa: BLE001
                pass
            model = YOLO(str(weights))

        # warm-up so the first real frame isn't slow
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=imgsz, verbose=False)
        self.model, self.imgsz, self.backend = model, imgsz, backend
        log.info("Model ready: %s via %s", preset["label"], backend)

    # ---------------- main loop -------------------------------------------- #
    def _set_status(self, status, msg=""):
        self.status = status
        self.error = msg if status in ("error", "loading", "waiting") else ""

    def _open_source(self):
        src_cfg = self.cfg.get("source", "")
        src, live = parse_source(src_cfg)
        if not live:
            p = Path(src)
            if not p.is_absolute():
                p = self.base / p
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p.name}. Upload a video from the dashboard.")
            src = str(p)
        obj = LiveSource(src) if live else FileSource(src, loop=self.cfg.get("loop", True))
        log.info("Opened source %s (%.1f fps)", src_cfg, obj.fps)
        return obj

    def _reset_tracker(self):
        pred = getattr(self.model, "predictor", None)
        for t in getattr(pred, "trackers", []) or []:
            try:
                t.reset()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self.tracks.clear()

    def _loop(self):
        self._source_obj = None
        self._proc_fps = 0.0
        self._infer_ms = 0.0
        self._analysed_pct = 100.0
        self._active_tracks = 0
        frame_no = 0
        last_t = time.perf_counter()
        processed_since, skipped_ref = 0, 0

        while self._running:
            try:
                if self._reload_model:
                    self._reload_model = False
                    self._load_model()
                    self._reset_tracker()

                if self._reload_source or self._source_obj is None:
                    self._reload_source = False
                    if self._source_obj:
                        self._source_obj.release()
                        self._source_obj = None
                    self._source_obj = self._open_source()
                    self._reset_tracker()
                    skipped_ref = 0

                if self.paused:
                    self._set_status("paused")
                    time.sleep(0.1)
                    self._source_obj.resume()
                    continue

                frame, vtime = self._source_obj.read()
                if frame is None:
                    raise IOError("Video source stopped sending frames")
                if self._source_obj.looped:
                    self._reset_tracker()

                frame_no += 1
                out = self._process(frame, frame_no, vtime)
                self._publish(out)
                self._set_status("running")

                # statistics
                processed_since += 1
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    inst = processed_since / (now - last_t)
                    self._proc_fps = inst if self._proc_fps == 0 else 0.7 * self._proc_fps + 0.3 * inst
                    skipped = self._source_obj.skipped - skipped_ref
                    skipped_ref = self._source_obj.skipped
                    total = processed_since + skipped
                    self._analysed_pct = 100.0 * processed_since / total if total else 100.0
                    processed_since, last_t = 0, now

            except Exception as e:  # noqa: BLE001 - the loop must never die
                log.exception("Engine error")
                self._set_status("error", str(e))
                self._proc_fps, self._active_tracks = 0.0, 0
                self._publish(self._message_frame(str(e)))
                if self._source_obj:
                    try:
                        self._source_obj.release()
                    except Exception:  # noqa: BLE001
                        pass
                    self._source_obj = None
                if self.model is None:
                    self._reload_model = True
                for _ in range(30):  # retry in 3 s (or sooner if settings change)
                    if not self._running or self._reload_source or self._reload_model:
                        break
                    time.sleep(0.1)

        if self._source_obj:
            self._source_obj.release()

    # ---------------- per-frame processing --------------------------------- #
    def _process(self, frame, frame_no, vtime):
        h, w = frame.shape[:2]
        if w > MAX_PROCESS_WIDTH:
            scale = MAX_PROCESS_WIDTH / w
            frame = cv2.resize(frame, (MAX_PROCESS_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        conf = float(self.cfg.get("confidence", 0.25))
        t0 = time.perf_counter()
        res = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            classes=list(VEHICLE_CLASSES),
            conf=max(0.05, conf * 0.4),  # low-score boxes feed ByteTrack's 2nd association
            iou=0.5,
            imgsz=self.imgsz,
            agnostic_nms=True,  # stops one vehicle getting both a "car" and a "truck" box
            verbose=False,
        )[0]
        dt = (time.perf_counter() - t0) * 1000
        self._infer_ms = dt if self._infer_ms == 0 else 0.8 * self._infer_ms + 0.2 * dt

        lx1, ly1, lx2, ly2 = self.cfg.get("line", [0.05, 0.6, 0.95, 0.6])
        a, b = (lx1 * w, ly1 * h), (lx2 * w, ly2 * h)

        seen = []
        boxes = res.boxes
        if boxes is not None and boxes.id is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            with self._lock:
                for box, tid, c, cf in zip(xyxy, ids, clss, confs):
                    name = VEHICLE_CLASSES.get(int(c))
                    if name is None:
                        continue
                    st = self.tracks.get(tid)
                    if st is None:
                        st = self.tracks[tid] = TrackState()
                    st.votes[name] += float(cf)
                    st.hits += 1
                    st.last_seen = frame_no
                    st.box = box if st.box is None else 0.55 * box + 0.45 * st.box
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    prev = st.trail[-1] if st.trail else None
                    st.trail.append((cx, cy))
                    if st.hits >= MIN_HITS_TO_COUNT and cf >= conf * 0.6:
                        self.unique_ids.add(tid)

                    side = _side_of_line((cx, cy), a, b)
                    if (not st.counted and prev is not None and st.side != 0 and side != 0
                            and side != st.side and st.hits >= MIN_HITS_TO_COUNT
                            and _segment_intersect(prev, (cx, cy), a, b)):
                        direction = "in" if st.side < 0 < side else "out"
                        st.counted = True
                        st.flash = 12
                        label = st.label
                        self.counts[direction][label] += 1
                        self._record_event(tid, label, direction, vtime)
                    if side != 0:
                        st.side = side
                    seen.append(tid)

                # forget tracks that disappeared a while ago
                for tid in [t for t, s in self.tracks.items() if frame_no - s.last_seen > 300]:
                    del self.tracks[tid]
                self._active_tracks = len(seen)

        return self._draw(frame, seen, a, b)

    def _record_event(self, tid, label, direction, vtime):
        ev = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "video_time": f"{int(vtime // 60):02d}:{vtime % 60:04.1f}" if vtime is not None else "",
            "id": int(tid),
            "type": label,
            "direction": direction,
        }
        self.events.append(ev)
        try:
            path = self.export_csv_path()
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=["date", *ev.keys(), "source"])
                if new:
                    wr.writeheader()
                wr.writerow({"date": datetime.now().strftime("%Y-%m-%d"), **ev,
                             "source": self.cfg.get("source")})
        except OSError as e:
            log.warning("Could not write crossings.csv: %s", e)

    # ---------------- drawing ---------------------------------------------- #
    def _draw(self, frame, seen, a, b):
        h, w = frame.shape[:2]
        th = max(1, round(w / 640))
        font_scale = 0.45 * th

        # counting line
        a_i, b_i = tuple(map(int, a)), tuple(map(int, b))
        cv2.line(frame, a_i, b_i, (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.line(frame, a_i, b_i, (0, 230, 255), th + 1, cv2.LINE_AA)

        with self._lock:
            for tid in seen:
                st = self.tracks.get(tid)
                if st is None or st.box is None:
                    continue
                x1, y1, x2, y2 = map(int, st.box)
                label = st.label
                color = (80, 255, 80) if st.flash > 0 else CLASS_COLORS.get(label, (255, 255, 255))
                if st.flash > 0:
                    st.flash -= 1
                if len(st.trail) > 1:
                    pts = np.array(st.trail, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, color, max(1, th), cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, th + 1, cv2.LINE_AA)
                text = f"{label} #{tid}"
                (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, th)
                ty = max(tht + 6, y1)
                cv2.rectangle(frame, (x1, ty - tht - 6), (x1 + tw + 6, ty), color, -1)
                cv2.putText(frame, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (20, 20, 20), th, cv2.LINE_AA)

            tin, tout = sum(self.counts["in"].values()), sum(self.counts["out"].values())

        # small HUD
        hud = f"IN {tin}   OUT {tout}   FPS {self._proc_fps:.0f}"
        (tw, tht), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.3, th + 1)
        cv2.rectangle(frame, (8, 8), (8 + tw + 16, 8 + tht + 16), (15, 15, 15), -1)
        cv2.putText(frame, hud, (16, 8 + tht + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 1.3, (255, 255, 255), th + 1, cv2.LINE_AA)
        return frame

    def _message_frame(self, text: str):
        img = np.full((540, 960, 3), 24, dtype=np.uint8)
        cv2.putText(img, "VisionTrack AI", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 200, 60), 3, cv2.LINE_AA)
        y = 170
        words, line = text.split(), ""
        for word in words:  # simple word-wrap
            if len(line) + len(word) > 55:
                cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (230, 230, 230), 2, cv2.LINE_AA)
                y, line = y + 40, ""
            line += word + " "
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        return img

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._frame_seq += 1
            self._cond.notify_all()
