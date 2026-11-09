from ._stub import StubAnalyzer


class OcclusionAnalyzer(StubAnalyzer):
    """카메라 가림 (얼굴 미검출 + 프레임 밝기/분산)."""
    name = "occlusion"
    requires = ()
