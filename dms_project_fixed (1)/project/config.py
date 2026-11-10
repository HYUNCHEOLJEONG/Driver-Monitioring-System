"""전역 설정. 임계값/경로/활성 분석기는 여기서만 바꾼다."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    face_model: Path = ROOT / "models" / "face_landmarker.task"
    yolo_model: Path = ROOT / "models" / "SOMKE_PHONE_MODEL_V1.pt"
    object_max_gap_ms: int = 2500
    driver_roi: tuple = (0.0, 0.0, 1.0, 1.0)  # normalized x1,y1,x2,y2
    camera_index: int = 0        # 외부 NIR 카메라 연결 시 변경
    frame_width: int = 640
    frame_height: int = 480
    yolo_conf: float = 0.25
    yolo_imgsz: int = 960
    smoking_hold_ms: int = 1000
    smoking_min_samples: int = 3
    smoking_mouth_distance: float = 0.20
    yolo_class_conf: dict = field(default_factory=lambda: {0: 0.4, 1: 0.6})
    yolo_every_n_frames: int = 3   # YOLO는 무거우므로 N프레임마다 실행(사이엔 직전 결과 재사용)
    yolo_classes: tuple = ()  # phone ID is resolved from checkpoint names
    draw_landmarks: bool = True
    
    # [수정] 활성화할 분석기 목록에 "fatigue" 추가
    enabled_analyzers: tuple = (
        "eye_state", "eye_closure", "perclos", "gaze", "fatigue",
    )
    
    # 분석기별 파라미터 (분석기 이름 -> dict)
    analyzer_params: dict = field(default_factory=lambda: {
        "eye_state": {"ear_threshold": 0.21, "calibration_sec": 3.0, "closed_ratio": 0.65},
        "eye_closure": {"warn_sec": 1.0, "alarm_sec": 2.0, "beep": True},
        "perclos": {"window_sec": 30.0, "warn": 0.15, "alarm": 0.30, "min_coverage": 0.5},
        "gaze": {},
        "fatigue": {"fatigue_threshold": 30.0},  # [추가] 테스트를 위해 임계치 설정 (기본 100점은 너무 높아서 낮게 잡음)
    })