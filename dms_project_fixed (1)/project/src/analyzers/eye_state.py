from ._stub import StubAnalyzer


class EyeStateAnalyzer(StubAnalyzer):
    """눈 개폐(EAR). 정규화 좌표 -> 픽셀 변환 후 계산(종횡비 불변)."""
    name = "eye_state"
    requires = ()
