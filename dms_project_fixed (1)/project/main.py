import time
import cv2
from config import Config
from src.face_detector import FaceDetector
from src.object_detector import ObjectDetector
from src.pipeline import Pipeline


def open_camera(cfg):
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

        # 리스트 형태의 analyzers에서 "gaze" 분석기를 안전하게 찾아내는 함수
        def get_gaze_analyzer():
            if hasattr(pipeline, "analyzers"):
                if isinstance(pipeline.analyzers, dict):
                    return pipeline.analyzers.get("gaze")
                elif isinstance(pipeline.analyzers, list):
                    for a in pipeline.analyzers:
                        if getattr(a, "name", None) == "gaze":
                            return a
            return None

        def mouse_callback(event, x, y, flags, param):
            gaze_analyzer = get_gaze_analyzer()
            if gaze_analyzer and hasattr(gaze_analyzer, "handle_mouse_click"):
                gaze_analyzer.handle_mouse_click(event, x, y)

        window_name = "Driver Monitoring System"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

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
                
                face_count = len(ctx.face_result.face_landmarks) if (ctx.face_result and ctx.face_result.face_landmarks) else 0
                cv2.putText(frame, f"Faces: {face_count}  FPS: {fps:.1f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("b") or key == ord("B"):
                    # 키보드 'B'를 눌러도 3분 바이패스 토글 가능
                    gaze_analyzer = get_gaze_analyzer()
                    if gaze_analyzer and hasattr(gaze_analyzer, "toggle_bypass"):
                        gaze_analyzer.toggle_bypass()
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        face.close()


if __name__ == "__main__":
    main()