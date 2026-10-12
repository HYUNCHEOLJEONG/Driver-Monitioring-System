from ..base import Analyzer
from ..types import FrameContext, AnalyzerResult


class StubAnalyzer(Analyzer):
    """미구현 분석기용. 파이프라인은 정상 동작하고 결과는 'not_implemented'."""

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        # TODO: 구현
        return AnalyzerResult(self.name, value=None, info={"status": "not_implemented"})
