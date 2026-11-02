"""새 분석기 추가 방법: 파일 생성 -> Analyzer 상속 -> 아래 REGISTRY에 한 줄 -> config.enabled_analyzers에 이름 추가."""
from .eye_state import EyeStateAnalyzer
from .perclos import PerclosAnalyzer
from .eye_closure import EyeClosureAnalyzer
from .microsleep import MicrosleepAnalyzer
from .gaze import GazeAnalyzer
from .distraction import DistractionAnalyzer
from .occlusion import OcclusionAnalyzer

REGISTRY = {c.name: c for c in (
    EyeStateAnalyzer, PerclosAnalyzer, EyeClosureAnalyzer, MicrosleepAnalyzer,
    GazeAnalyzer, DistractionAnalyzer, OcclusionAnalyzer,
)}
