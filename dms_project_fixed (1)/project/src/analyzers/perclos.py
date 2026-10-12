from ._stub import StubAnalyzer


class PerclosAnalyzer(StubAnalyzer):
    """슬라이딩 윈도우(기본 30s) 눈감김 비율. 주의/경보 판정."""
    name = "perclos"
    requires = ('eye_state',)
