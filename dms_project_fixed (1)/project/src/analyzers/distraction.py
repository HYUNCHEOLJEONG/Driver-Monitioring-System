from ._stub import StubAnalyzer


class DistractionAnalyzer(StubAnalyzer):
    """휴대폰/흡연 검출 (ctx.objects 사용)."""
    name = "distraction"
    requires = ()
