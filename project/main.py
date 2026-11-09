import sys
import time
import cv2
import numpy as np
from config import Config
from src.face_detector import FaceDetector
from src.object_detector import ObjectDetector
from src.pipeline import Pipeline


def list_cameras(max_index=6):
    for i in range(max_index):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cam.read() if cam.isOpened() else (False, None)
        print(f"index {i}: {'OK ' + str(frame.shape) if ok else '-'}")
        cam.release()


def open_camera(cfg):
    # Windows: 기본 MSMF는 느리게 열리거나 실패하는 경우가 많아 DSHOW 우선
    cam = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(cfg.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.camera_index}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    return cam


def main():
    if "--list-cameras" in sys.argv:
        list_cameras()
        return
    cfg = Config()
    if "--camera" in sys.argv:                      # 예: python main.py --camera 1
        cfg.camera_index = int(sys.argv[sys.argv.index("--camera") + 1])
    face = FaceDetector(cfg.face_model)
    try:
        obj = ObjectDetector(cfg.yolo_model, cfg.yolo_conf, cfg.yolo_classes)
        pipeline = Pipeline(face, obj, cfg)
        camera = open_camera(cfg)
        try:
            start = previous = time.monotonic()
            last_ts = -1
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Failed to read camera frame")
                    break
                if frame.ndim == 2 or frame.shape[2] == 1:   # NIR 흑백 영상 -> 3채널
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = ts

                ctx = pipeline.process(frame, ts)
                if cfg.draw_landmarks:
                    face.draw(frame, ctx.face_result)
                obj.draw(frame, ctx.objects)
                pipeline.draw(ctx)

                now = time.monotonic()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                cv2.putText(frame, f"Faces: {len(ctx.face_result.face_landmarks)}  FPS: {fps:.1f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Driver Monitoring System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                pipeline.on_key(key)      # vehicle.source == "manual" 일 때 r/a/d/h
        finally:
            pipeline.close()
            camera.release()
            cv2.destroyAllWindows()
    finally:
        face.close()


if __name__ == "__main__":
    main()
