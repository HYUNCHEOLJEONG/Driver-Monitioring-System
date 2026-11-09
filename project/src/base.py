from abc import ABC, abstractmethod
from .types import FrameContext, AnalyzerResult


class Analyzer(ABC):
    """모든 판정 모듈의 공통 인터페이스.

    name: 레지스트리/설정에서 쓰는 고유 이름
    requires: 먼저 실행되어야 하는 분석기 이름들 (예: perclos -> eye_state)
    """
    name: str = ""
    requires: tuple = ()

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def update(self, ctx: FrameContext) -> AnalyzerResult:
        """프레임 1장 처리. 상태(슬라이딩 윈도우 등)는 인스턴스에 보관."""

    def draw(self, frame, result: AnalyzerResult):
        """선택: 결과 오버레이."""
        return frame

    def reset(self):
        """선택: 상태 초기화."""
