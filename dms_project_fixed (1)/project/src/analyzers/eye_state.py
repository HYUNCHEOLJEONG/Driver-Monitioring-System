from collections import deque
import numpy as np
from ..base import Analyzer
from ..types import FrameContext, AnalyzerResult
from ..geometry import landmarks_to_pixels, eye_aspect_ratio

# MediaPipe FaceMesh 인덱스 [외측 안각, 위1, 위2, 내측 안각, 아래2, 아래1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # 화면 기준 왼쪽 눈(사용자의 오른쪽 눈)
LEFT_EYE = [362, 385, 387, 263, 373, 380]


class EyeStateAnalyzer(Analyzer):
    """눈 개폐. 정규화 좌표 -> 픽셀 좌표 변환 후 EAR 계산(종횡비 불변).

    value = 양안 평균 EAR (얼굴 미검출 시 None)
    info  = ear_left/right, closed(bool|None), threshold, face(bool),
            blink_score(MediaPipe blendshape eyeBlink 평균, 보조 신호), calibrated

    params:
      ear_threshold   고정 임계값 (calibration_sec=0 이거나 보정 실패 시 사용)
      calibration_sec 시작 후 이 시간 동안 '뜬 눈' EAR 기준선을 수집 (0이면 끔)
      closed_ratio    보정 사용 시 임계값 = 기준선 * closed_ratio
    """
    name = "eye_state"
    requires = ()

    def __init__(self, ear_threshold=0.21, calibration_sec=3.0, closed_ratio=0.7, **kw):
        super().__init__(**kw)
        self.fixed_thr = ear_threshold
        self.calib_ms = calibration_sec * 1000
        self.ratio = closed_ratio
        self.reset()

    def reset(self):
        self.threshold = self.fixed_thr
        self._t0 = None
        self._samples = []
        self._calibrated = False

    def _blink_score(self, ctx):
        try:
            bs = ctx.face_result.face_blendshapes[0]
            d = {c.category_name: c.score for c in bs}
            return (d["eyeBlinkLeft"] + d["eyeBlinkRight"]) / 2
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    def _calibrate(self, ctx, ear):
        if self.calib_ms <= 0 or self._calibrated:
            return
        if self._t0 is None:
            self._t0 = ctx.timestamp_ms
        self._samples.append(ear)
        if ctx.timestamp_ms - self._t0 >= self.calib_ms:
            # 깜빡임 영향 제거: 상위 50%의 중앙값 = 뜬 눈 EAR
            s = np.sort(self._samples)
            base = float(np.median(s[len(s) // 2:]))
            if base > 0.10:   # 비정상 기준선(눈 감고 시작 등)이면 고정값 유지
                self.threshold = base * self.ratio
            self._calibrated = True

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        faces = getattr(ctx.face_result, "face_landmarks", None)
        if not faces:
            return AnalyzerResult(self.name, None, info={"face": False, "closed": None})
        pts = landmarks_to_pixels(faces[0], ctx.width, ctx.height)
        el = eye_aspect_ratio(pts[LEFT_EYE])
        er = eye_aspect_ratio(pts[RIGHT_EYE])
        ear = (el + er) / 2
        self._calibrate(ctx, ear)
        closed = ear < self.threshold
        return AnalyzerResult(self.name, ear, info={
            "face": True, "ear_left": el, "ear_right": er, "closed": closed,
            "threshold": self.threshold, "calibrated": self._calibrated,
            "blink_score": self._blink_score(ctx),
        })

    def draw(self, frame, result):
        import cv2
        i = result.info
        if not i.get("face"):
            return frame
        color = (0, 0, 255) if i["closed"] else (0, 255, 0)
        tag = "" if i["calibrated"] or self.calib_ms <= 0 else " (calibrating)"
        cv2.putText(frame, f"EAR {result.value:.2f} / thr {i['threshold']:.2f}{tag}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
