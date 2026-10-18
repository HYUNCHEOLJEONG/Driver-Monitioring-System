from pathlib import Path
import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, confidence=0.4, classes=None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.classes = list(classes) if classes else None  # None = 전체 클래스

    def detect(self, frame):
        result = self.model.predict(source=frame, conf=self.confidence, imgsz=640, classes=self.classes, verbose=False)[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "class_id": class_id,
                "label": self.model.names[class_id],
                "confidence": float(box.conf[0].item()),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
            })
        return detections

    @staticmethod
    def draw(frame, detections):
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            text = f'{obj["label"]} {obj["confidence"]:.2f}'
            cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
        return frame
