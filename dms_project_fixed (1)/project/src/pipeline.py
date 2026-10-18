import time
from .types import FrameContext
from .analyzers import REGISTRY


class Pipeline:
    """검출기 실행 -> 분석기 의존 순서대로 실행."""

    def __init__(self, face_detector, object_detector, config):
        self.face = face_detector
        self.obj = object_detector
        self.cfg = config
        self.analyzers = self._build(config)
        self._frame_idx = 0
        self._last_objects = []

    @staticmethod
    def _build(config):
        names = list(config.enabled_analyzers)
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise KeyError(f"Unknown analyzers: {unknown}. Available: {list(REGISTRY)}")
        ordered, seen = [], set()

        def visit(n):  # 의존성 위상 정렬
            if n in seen:
                return
            seen.add(n)
            for r in REGISTRY[n].requires:
                visit(r)
            ordered.append(n)

        for n in names:
            visit(n)
        return [REGISTRY[n](**config.analyzer_params.get(n, {})) for n in ordered]

    def process(self, frame, timestamp_ms):
        h, w = frame.shape[:2]
        ctx = FrameContext(frame=frame, timestamp_ms=timestamp_ms, width=w, height=h)
        ctx.face_result = self.face.detect(frame, timestamp_ms)
        if self._frame_idx % max(1, self.cfg.yolo_every_n_frames) == 0:
            self._last_objects = self.obj.detect(frame)
        ctx.objects = self._last_objects
        self._frame_idx += 1
        for a in self.analyzers:
            ctx.results[a.name] = a.update(ctx)
        return ctx

    def draw(self, ctx):
        for a in self.analyzers:
            a.draw(ctx.frame, ctx.results[a.name])
        return ctx.frame
