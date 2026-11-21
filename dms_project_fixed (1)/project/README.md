# Driver Monitoring System - Detection Prototype

Uses OpenCV webcam capture, MediaPipe Face Landmarker (VIDEO mode), and Ultralytics YOLO11n.

## Setup

Use a compatible Python environment (Python 3.10 or 3.11 recommended).

```powershell
python -m pip install -r requirements.txt
```

Download and save model files as:

- `models/face_landmarker.task` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- `models/yolo11n.pt` from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt (or use Ultralytics' official auto-download via `YOLO('yolo11n.pt')` and move it into models/).

## Run

```powershell
python main.py
```

Press `q` to quit. Change `CAMERA_INDEX` in `main.py` when connecting a USB/NIR camera.

The model weights are not bundled in this source archive. The model output and FPS depend on hardware. YOLO11n pretrained COCO weights recognize cell phones but do not identify smoking as a dedicated class.
