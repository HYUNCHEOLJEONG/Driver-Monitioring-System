"""전역 설정. 임계값/경로/활성 분석기는 여기서만 바꾼다."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    face_model: Path = ROOT / "models" / "face_landmarker.task"
    yolo_model: Path = ROOT / "models" / "yolo11n.pt"
    camera_index: int = 0          # 외부 NIR 카메라 연결 시 변경
    frame_width: int = 640
    frame_height: int = 480
    yolo_conf: float = 0.4
    yolo_every_n_frames: int = 3   # YOLO는 무거우므로 N프레임마다 실행(사이엔 직전 결과 재사용)
    yolo_classes: tuple = (67,)    # COCO: 67=cell phone. 흡연은 전용 클래스 없음 -> 추후 별도 모델
    draw_landmarks: bool = True
    # 활성화할 분석기 이름 (src/analyzers/__init__.py 의 REGISTRY 키)
    enabled_analyzers: tuple = (
        "eye_state", "eye_closure", "perclos", "microsleep", "gaze", "distraction", "occlusion",
    )
    # 분석기별 파라미터 (분석기 이름 -> dict)
    analyzer_params: dict = field(default_factory=lambda: {
        "eye_state": {"ear_threshold": 0.21, "calibration_sec": 3.0, "closed_ratio": 0.65},
        "eye_closure": {"warn_sec": 1.0, "alarm_sec": 2.0, "beep": True},
        "perclos": {"window_sec": 30.0, "warn": 0.15, "alarm": 0.30, "min_coverage": 0.5},
        "microsleep": {},
        "gaze": {},
        "distraction": {},
        "occlusion": {},
    })

