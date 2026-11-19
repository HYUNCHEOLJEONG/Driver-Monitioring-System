import threading
import numpy as np
from ..base import Analyzer
from ..types import FrameContext, AnalyzerResult
from ..geometry import landmarks_to_pixels

try:
    import winsound  # Windows 전용
except ImportError:
    winsound = None

# MediaPipe FaceMesh 입 안쪽 입술 랜드마크
UP_IN, LOW_IN = 13, 14        # 안쪽 윗입술 / 아랫입술 중앙
L_CORNER, R_CORNER = 78, 308  # 입꼬리


def mouth_aspect_ratio(pts):
    """MAR = 입술 안쪽 세로 간격 / 입꼬리 가로 폭. 다문 입 ≈ 0, 말할 때 ≈ 0.1~0.4, 하품 ≈ 0.6+."""
    v = np.linalg.norm(pts[UP_IN] - pts[LOW_IN])
    h = np.linalg.norm(pts[L_CORNER] - pts[R_CORNER])
    return float(v / (h + 1e-6))


class YawnAnalyzer(Analyzer):
    """하품 감지: 입을 크게 벌린 상태가 일정 시간 이어지면 하품.

    제외 규칙
      - 대화: 입이 작게 벌어짐(MAR < open_mar) -> 하품 후보가 되지 못함
      - 뻐끔거림: 크게 벌려도 close_grace_ms 넘게 닫히면 타이머 리셋 -> min_sec 못 채움
    params:
      open_mar        '크게 벌림' 기준 (기본 0.60). 말할 때 값(0.1~0.4)보다 높게
      min_sec         크게 벌린 채 유지해야 하는 시간 (기본 1.5s)
      close_grace_ms  이 시간보다 짧은 입 다묾/MAR 흔들림은 유지로 간주 (기본 250ms)
      lost_grace_ms   얼굴 미검출 시 타이머를 유지하는 시간 (기본 300ms)
      hold_sec        하품 판정 후 경고 표시를 유지하는 시간 (기본 3s)
      count_window_sec / count_alarm  이 시간 내 하품 횟수가 count_alarm 이상이면 alarm (기본 600s / 3회)
      beep            경고음
    value = 현재 입을 크게 벌린 지속 시간(초)
    """
    name = "yawn"
    requires = ()

    def __init__(self, open_mar=0.60, min_sec=1.5, close_grace_ms=250, lost_grace_ms=300,
                 hold_sec=3.0, count_window_sec=600.0, count_alarm=3, beep=True, **kw):
        super().__init__(**kw)
        self.open_mar, self.min_ms = open_mar, min_sec * 1000
        self.close_grace, self.lost_grace = close_grace_ms, lost_grace_ms
        self.hold_ms, self.win_ms, self.count_alarm = hold_sec * 1000, count_window_sec * 1000, count_alarm
        self.beep = beep
        self.reset()

    def reset(self):
        self._start = None
        self._last_open = None
        self._last_seen = None
        self._counted = False     # 현재 벌림에서 이미 하품으로 센 경우
        self._hold_until = -10**9
        self._events = []         # 하품 판정 시각들

    def _beep(self, freq):
        if self.beep and winsound:
            threading.Thread(target=winsound.Beep, args=(freq, 300), daemon=True).start()

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        t = ctx.timestamp_ms
        faces = getattr(ctx.face_result, "face_landmarks", None)
        mar, face = None, bool(faces)

        if face:
            self._last_seen = t
            mar = mouth_aspect_ratio(landmarks_to_pixels(faces[0], ctx.width, ctx.height))
            if mar >= self.open_mar:
                if self._start is None:
                    self._start = t
                self._last_open = t
            elif self._start is not None and t - self._last_open > self.close_grace:
                self._start, self._counted = None, False     # 닫힘/작은 벌림 -> 리셋(대화·뻐끔 제외)
        elif self._start is not None and t - self._last_seen > self.lost_grace:
            self._start, self._counted = None, False

        dur = (t - self._start) if self._start is not None else 0
        if self._start is not None and not self._counted and dur >= self.min_ms:
            self._counted = True
            self._events.append(t)
            self._hold_until = t + self.hold_ms
            self._beep(1000)
        self._events = [e for e in self._events if t - e <= self.win_ms]

        count = len(self._events)
        if t < self._hold_until:
            alert = "alarm" if count >= self.count_alarm else "warning"
        else:
            alert = "none"
        state = ("yawning" if self._start is not None and dur >= self.min_ms else
                 "opening" if self._start is not None else
                 "talking" if mar is not None and mar >= 0.2 else "closed" if face else "no_face")
        return AnalyzerResult(self.name, dur / 1000, alert,
                              {"face": face, "mar": mar, "state": state, "count": count})

    def draw(self, frame, result):
        import cv2
        i = result.info
        if not i.get("face"):
            return frame
        h, w = frame.shape[:2]
        color = {"none": (0, 255, 0), "warning": (0, 165, 255), "alarm": (0, 0, 255)}[result.alert]
        cv2.putText(frame, f"MAR {i['mar']:.2f} [{i['state']}] yawns {i['count']}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if result.alert != "none":
            msg = "FREQUENT YAWNING" if result.alert == "alarm" else "YAWN DETECTED"
            cv2.putText(frame, msg, (20, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return frame
