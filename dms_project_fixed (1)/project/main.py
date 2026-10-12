import time
import cv2
from config import Config
from src.face_detector import FaceDetector
from src.object_detector import ObjectDetector
from src.pipeline import Pipeline


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
    cfg = Config()
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        face.close()


if __name__ == "__main__":
    main()
