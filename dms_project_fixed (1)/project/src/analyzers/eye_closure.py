from ..warning_rules import closure_limit
from ..base import Analyzer
from ..types import FrameContext, AnalyzerResult



class EyeClosureAnalyzer(Analyzer):
    """연속 눈 감김 시간 감지. PERCLOS(30초 누적)가 못 잡는 '순간 졸음'용 즉시 경고.

    value = 현재 연속 감김 시간(초)
    alert = none | alarm (기본 2초, 80km/h 이상 1초)

    params:
      warn_sec        이전 호출 호환용. 현재 판정에는 사용하지 않음
      alarm_sec       이전 호출 호환용. 실제 기준은 warning_rules에서 선택
      open_grace_ms   이 시간보다 짧게 눈이 떠지는 노이즈는 감김 지속으로 간주 (기본 150ms)
      lost_grace_ms   얼굴 미검출 시 감김 타이머를 유지하는 시간 (기본 500ms, 고개 떨굼 대비)
      beep            Windows 경고음 사용 여부
      beep_interval_s 경보음 반복 간격
    """
    name = "eye_closure"
    requires = ("eye_state",)

    def __init__(self, warn_sec=1.0, alarm_sec=2.0, open_grace_ms=150,
                 lost_grace_ms=500, beep=True, beep_interval_s=1.0, **kw):
        super().__init__(**kw)
        self.warn_ms, self.alarm_ms = warn_sec * 1000, alarm_sec * 1000
        self.open_grace, self.lost_grace = open_grace_ms, lost_grace_ms
        self.beep, self.beep_gap = beep, beep_interval_s * 1000
        self.reset()

    def reset(self):
        self._start = None       # 감김 시작 시각
        self._last_closed = None # 마지막으로 '감김'이 확인된 시각
        self._last_seen = None   # 마지막으로 얼굴이 보인 시각
        self._last_beep = -10**9

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        t = ctx.timestamp_ms
        es = ctx.results["eye_state"].info
        face = es.get("face", False)

        if face:
            self._last_seen = t
            if es["closed"]:
                if self._start is None:
                    self._start = t
                self._last_closed = t
            elif self._start is not None and t - self._last_closed > self.open_grace:
                self._start = None            # 눈이 충분히 떠짐 -> 해제
        elif self._start is not None and t - self._last_seen > self.lost_grace:
            self._start = None                # 얼굴을 오래 잃음 -> 해제(가림/이탈은 다른 모듈)

        dur = (t - self._start) if self._start is not None else 0
        limit = closure_limit(ctx.speed_kmh)
        alert = "alarm" if dur >= limit * 1000 else "none"
        return AnalyzerResult(self.name, dur / 1000, alert,
                              {"face": face, "alarm_sec": limit, "beep": self.beep,
                               "beep_interval_ms": self.beep_gap})

    def draw(self, frame, result):
        import cv2
        if result.alert == "none":
            return frame
        h, w = frame.shape[:2]
        color = (0, 0, 255) if result.alert == "alarm" else (0, 165, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
        msg = "WAKE UP! EYES CLOSED" if result.alert == "alarm" else "DROWSY: EYES CLOSED"
        cv2.putText(frame, f"{msg} {result.value:.1f}s", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return frame
