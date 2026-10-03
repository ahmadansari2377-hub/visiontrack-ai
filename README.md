# VisionTrack AI v2

Real-time vehicle detection, tracking and counting in your browser.

## Start it

1. Double-click **`run.bat`**.
   - The first time, it sets everything up by itself (5-10 minutes, needs internet).
   - After that, it starts in a few seconds.
2. Your browser opens at http://127.0.0.1:5000 automatically.
3. Keep the black window open while you use the app. Close it to stop.

Optional: double-click **`create_desktop_shortcut.bat`** once to get a "VisionTrack AI" icon on your
Desktop. Unlike the old `.url` shortcut, it starts the server too.

## Add your video

Either:
- use **Upload a video** on the dashboard, or
- copy the video file into the `videos` folder, then pick it under **Saved videos** and click **Play**.

For a webcam, type `0` in the **Webcam or stream** box and click **Connect**. For an IP camera, paste its `rtsp://...` address instead.

## Using the dashboard

| Control | What it does |
|---|---|
| **Set counting line** | Click two points on the video. Vehicles are counted when their centre crosses this line. Put it across the road where cars are clearly visible. |
| **Model** | **Fast** keeps up with the video on almost any laptop. **Balanced** and **Accurate** find smaller and more distant vehicles but analyse fewer frames per second. |
| **Confidence** | Lower it (e.g. 0.20) if vehicles are missed. Raise it (e.g. 0.45) if you see boxes on things that aren't vehicles. |
| **Download CSV** | Every counted vehicle, with the time, type and direction. |
| **Snapshot** | Saves the current frame as a picture. |

The numbers at the top are real measurements:
- **Processing fps** is how many frames per second the AI analyses.
- **Frames analysed** is the share of video frames checked. When the CPU can't keep up, the video still plays at normal speed and some frames are skipped. Counting stays correct because tracking handles the gaps.

## Why it's more reliable than v1

- **Boxes never lag behind the cars.** They're drawn on the same frame the AI looked at.
- **Correct speed.** The video plays at normal speed: no slow motion, no speed-up.
- **Intel OpenVINO acceleration.** On first start, the model is converted for your Intel CPU, which is usually 2-3x faster than plain PyTorch. If that fails, the app falls back to PyTorch automatically.
- **No double counts.** A vehicle is counted once. It must be tracked for 3+ frames and actually cross the line segment.
- **Stable vehicle types.** Each vehicle's type is decided by a vote over all its frames, so a car doesn't flicker between "car" and "truck". Overlapping car/truck boxes are merged.
- **Tuned ByteTrack** (`tracker.yaml`). A vehicle keeps its ID for about 2 seconds while it is hidden behind something.
- **Nothing crashes the server.** A missing video, an unplugged camera or a bad frame shows a message and retries. If the whole process ever stops, `run.bat` restarts it.
- **Settings are saved** (`config.json`): your line, model, confidence and video are remembered.
- **Logs** are in `logs/visiontrack.log` if something goes wrong.

## Files

```
app.py          web server + API
engine.py       detection, tracking, counting
tracker.yaml    tracker tuning
templates/      dashboard
run.bat         one-click start (setup + auto-restart)
videos/         your videos
models/         AI models (downloaded automatically)
logs/           log file + crossings.csv
```
