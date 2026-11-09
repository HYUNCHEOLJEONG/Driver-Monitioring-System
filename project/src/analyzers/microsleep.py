from ._stub import StubAnalyzer


class MicrosleepAnalyzer(StubAnalyzer):
    """고개 떨굼 + 하품(입 벌림 지속) 기반 초기 졸음."""
    name = "microsleep"
    requires = ('eye_state',)
