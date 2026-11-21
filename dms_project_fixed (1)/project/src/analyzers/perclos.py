from collections import deque
from ..base import Analyzer
from ..types import FrameContext, AnalyzerResult


class PerclosAnalyzer(Analyzer):
    """PERCLOS = 슬라이딩 윈도우(기본 30s) 내 눈 감김 시간 비율. 주의/경보 판정.

    프레임 수가 아니라 '시간 가중' 비율이라 FPS가 변해도 값이 일정하다.
    얼굴 미검출 구간은 분모에서 제외한다(가림/이탈은 occlusion 등 다른 모듈 담당).
    윈도우 내 유효 데이터가 min_coverage 미만이면 판정하지 않는다.

    value = PERCLOS (0~1) | None,  alert = none | warning | alarm
    """
    name = "perclos"
    requires = ("eye_state",)

    def __init__(self, window_sec=30.0, warn=0.15, alarm=0.30, min_coverage=0.5, **kw):
        super().__init__(**kw)
        self.window_ms = window_sec * 1000
        self.warn, self.alarm = warn, alarm
        self.min_coverage = min_coverage
        self.reset()

    def reset(self):
        self._buf = deque()      # (timestamp_ms, closed | None)
        self._last = None

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        es = ctx.results["eye_state"].info
        closed = es.get("closed") if es.get("face") else None
        self._buf.append((ctx.timestamp_ms, closed))
        cutoff = ctx.timestamp_ms - self.window_ms
        while len(self._buf) > 1 and self._buf[1][0] <= cutoff:
            self._buf.popleft()

        # 각 샘플은 '다음 샘플까지의 시간' 동안 유지된 상태로 가중
        closed_t = valid_t = 0.0
        items = list(self._buf)
        for (t0, c), (t1, _) in zip(items, items[1:]):
            dt = t1 - max(t0, cutoff)
            if dt <= 0 or c is None:
                continue
            valid_t += dt
            if c:
                closed_t += dt

        span = ctx.timestamp_ms - max(items[0][0], cutoff)
        # 윈도우가 아직 덜 찼어도 min_coverage 비율 이상 쌓이면 판정 시작
        enough = valid_t >= self.window_ms * self.min_coverage * min(1.0, 1.0) and valid_t > 0
        p = closed_t / valid_t if enough else None
        alert = "none"
        if p is not None:
            alert = "alarm" if p >= self.alarm else "warning" if p >= self.warn else "none"
        return AnalyzerResult(self.name, p, alert, {
            "valid_sec": valid_t / 1000, "window_sec": self.window_ms / 1000,
        })

    def draw(self, frame, result):
        import cv2
        if result.value is None:
            txt, color = f"PERCLOS -- (collecting {result.info['valid_sec']:.0f}s)", (200, 200, 200)
        else:
            color = {"none": (0, 255, 0), "warning": (0, 200, 255), "alarm": (0, 0, 255)}[result.alert]
            txt = f"PERCLOS {result.value*100:.1f}%  [{result.alert.upper()}]"
        cv2.putText(frame, txt, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
