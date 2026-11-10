"""새 분석기 추가 방법: 파일 생성 -> Analyzer 상속 -> 아래 REGISTRY에 한 줄 -> config.enabled_analyzers에 이름 추가."""
from .eye_state import EyeStateAnalyzer
from .perclos import PerclosAnalyzer
from .eye_closure import EyeClosureAnalyzer
from .microsleep import MicrosleepAnalyzer
from .gaze import GazeAnalyzer
from .distraction import DistractionAnalyzer
from .occlusion import OcclusionAnalyzer
from .fatigue_analyzer import FatigueAnalyzer  # [추가] 피로도 분석기 임포트

REGISTRY = {c.name: c for c in (
    EyeStateAnalyzer, PerclosAnalyzer, EyeClosureAnalyzer, MicrosleepAnalyzer,
    GazeAnalyzer, DistractionAnalyzer, OcclusionAnalyzer, FatigueAnalyzer,  # [추가] 튜플에 FatigueAnalyzer 포함
)}