import time
import cv2
from config import Config
from src.face_detector import FaceDetector
from src.object_detector import ObjectDetector
from src.pipeline import Pipeline
from src.camera_capture import open_camera


def main():
    cfg = Config()
    face = FaceDetector(cfg.face_model, num_faces=2)
    pipeline = None
    try:
        obj = ObjectDetector(cfg.yolo_model, cfg.yolo_conf, cfg.yolo_classes, cfg.yolo_class_conf, imgsz=cfg.yolo_imgsz)
        pipeline = Pipeline(face, obj, cfg)
        camera = open_camera(cfg)
        pipeline.events.emit("SESSION", f"모니터링 시작 · 카메라={cfg.camera_index} · 모델={cfg.yolo_model.name}")

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

        window_name = "Driver Monitoring System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 900)

        try:
            start = previous = time.monotonic()
            last_ts = -1
            preview_shown = False
            failed_frames = 0
            while True:
                if preview_shown and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                ok, frame = camera.read()
                if not ok or frame is None or not frame.size:
                    failed_frames += 1
                    if failed_frames == 1:
                        pipeline.events.emit("CAMERA", "프레임 수신 실패 · 재시도 중", "WARN")
                    if failed_frames >= 5:
                        print("Camera stopped delivering frames (5 consecutive failures). Check connection or other camera apps.")
                        break
                    if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                    continue
                if failed_frames:
                    pipeline.events.emit("CAMERA", "프레임 수신 복구")
                failed_frames = 0
                ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = ts

                ctx = pipeline.process(frame, ts)
                now = time.monotonic()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                
                cv2.imshow(window_name, pipeline.yolo_status.render(
                    ctx, fps=fps, camera_index=cfg.camera_index, draw_landmarks=cfg.draw_landmarks))
                preview_shown = True
                
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("r"), ord("R")):
                    gaze_analyzer = get_gaze_analyzer()
                    if gaze_analyzer is not None:
                        gaze_analyzer.reset()
                        pipeline.events.emit("GAZE", "전방 자세 재보정 시작 · 정면을 2초간 보세요")
                elif key == ord("b") or key == ord("B"):
                    # 키보드 'B'를 눌러도 3분 바이패스 토글 가능
                    gaze_analyzer = get_gaze_analyzer()
                    if gaze_analyzer and hasattr(gaze_analyzer, "toggle_bypass"):
                        gaze_analyzer.toggle_bypass()
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        if pipeline is not None:
            pipeline.events.emit("SESSION", "모니터링 종료")
            pipeline.close()
        face.close()


if __name__ == "__main__":
    main()