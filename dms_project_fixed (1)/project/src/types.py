"""파이프라인 전체가 공유하는 데이터 구조."""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class FrameContext:
    """한 프레임에 대한 입력 + 검출 결과. 모든 분석기가 이것만 받는다."""
    frame: np.ndarray                 # BGR
    timestamp_ms: int
    width: int
    height: int
    face_result: Any = None           # MediaPipe FaceLandmarkerResult
    objects: list = field(default_factory=list)   # ObjectDetector 결과(dict 리스트)
    results: dict = field(default_factory=dict)   # 앞선 분석기 결과 (name -> AnalyzerResult)


@dataclass
class AnalyzerResult:
    name: str
    value: Any = None                 # 수치/상태 (예: EAR, PERCLOS)
    alert: str = "none"               # none | warning | alarm
    info: dict = field(default_factory=dict)
