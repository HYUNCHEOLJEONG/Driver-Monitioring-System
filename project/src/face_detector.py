from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self, model_path, num_faces=1):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
        # 경로 대신 버퍼로 로드: 한글/비ASCII 경로(Windows)에서도 안전
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,                # 하품/입 벌림 등 추후 사용
            output_facial_transformation_matrixes=True,  # 고개 자세(pitch/yaw) 추후 사용
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, int(timestamp_ms))

    @staticmethod
    def draw(frame, result):
        h, w = frame.shape[:2]
        for landmarks in result.face_landmarks:
            for p in landmarks:
                x = max(0, min(w - 1, int(p.x * w)))
                y = max(0, min(h - 1, int(p.y * h)))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def close(self):
        self.landmarker.close()
