from pathlib import Path
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self,model_path,confidence=0.25,classes=None,class_confidence=None,imgsz=960):
        model_path=Path(model_path)
        if not model_path.is_file():raise FileNotFoundError(f'YOLO model not found: {model_path}')
        self.model=YOLO(str(model_path))
        self.confidence=confidence
        self.class_confidence=dict(class_confidence or {})
        self.latest_scores={}
        if self.model.names!={0:'cell_phone',1:'cigarette'}:
            raise ValueError(f'Expected cell_phone/cigarette model, got {self.model.names}')
        self.classes=[0,1];self.imgsz=imgsz
        print(f'Object model: {model_path}; names={self.model.names}; imgsz={imgsz}',flush=True)

    def detect(self,frame):
        threshold=min([self.confidence,*self.class_confidence.values()])
        result=self.model.predict(source=frame,conf=threshold,imgsz=self.imgsz,classes=self.classes,verbose=False)[0]
        self.latest_scores={};detections=[]
        for box in result.boxes:
            c=int(box.cls[0]);score=float(box.conf[0])
            self.latest_scores[c]=max(score,self.latest_scores.get(c,0))
            if score<self.class_confidence.get(c,self.confidence):continue
            detections.append(dict(class_id=c,label=self.model.names[c],confidence=score,bbox=[int(v) for v in box.xyxy[0].tolist()]))
        return detections

