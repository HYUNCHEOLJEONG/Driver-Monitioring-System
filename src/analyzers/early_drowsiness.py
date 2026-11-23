"""기존 분석 결과를 읽기만 하는 초기 졸음 분석기. 기존 판정/타이머를 수정하지 않는다."""
from collections import deque
from dataclasses import replace
import math
from ..base import Analyzer
from ..types import AnalyzerResult
from ..early_drowsiness_config import EarlyDrowsinessConfig
from ..geometry import landmarks_to_pixels
from ..warning_rules import pause_reason
from .yawn import mouth_aspect_ratio


def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _points(value, bands):
    return max((points for lower, points in bands if value >= lower), default=0.0)


def calculate_early_score(perclos, long_count, yawn_count, nod_count, config=None):
    """부작용 없는 계산 함수. 같은 눈 사건의 가중은 명시적인 배율만 적용한다."""
    c = config or EarlyDrowsinessConfig()
    p = _number(perclos)
    p = p if p is not None and 0 <= p <= 1 else 0.0
    parts = {"perclos": _points(p, c.perclos_points),
             "long_closure": _points(long_count, c.long_points),
             "yawn": _points(yawn_count, c.yawn_points),
             "nod": _points(nod_count, c.nod_points)}
    multiplier = max((m for lower, n, m in c.interactions
                      if p >= lower and long_count >= n), default=1.0)
    total = (parts["perclos"] + parts["long_closure"]) * multiplier + parts["yawn"] + parts["nod"]
    return round(total, 3), parts, multiplier


class EarlyDrowsinessAnalyzer(Analyzer):
    name = "early_drowsiness"
    requires = ("eye_state", "perclos", "gaze")

    def __init__(self, **params):
        super().__init__(**params)
        self.cfg = replace(EarlyDrowsinessConfig(), **params)
        c = self.cfg
        if c.threshold <= 0 or not 0 <= c.clear_threshold < c.threshold:
            raise ValueError("early_drowsiness: invalid score thresholds")
        if not 0 < c.min_coverage <= 1 or c.quality_window_sec <= 0 or c.max_gap_ms <= 0:
            raise ValueError("early_drowsiness: invalid quality settings")
        if c.confirm_sec < 0 or c.clear_sec < 0 or c.repeat_sec <= 0:
            raise ValueError("early_drowsiness: invalid timing")
        self.reset()

    def reset(self):
        self._last_t = None
        self._stream_start = None
        self._last_valid = False
        self._coverage = deque()
        self._long_events, self._yawn_events, self._nod_events = deque(), deque(), deque()
        self._active = False
        self._high_since = self._low_since = None
        self._break_candidates()

    def _break_candidates(self):
        # 관측 공백을 눈감김·하품·끄덕임 사건으로 연결하지 않는다.
        self._eye_start = None
        self._eye_armed = False
        self._mouth_start = self._mouth_low = None
        self._mouth_counted = False
        self._mouth_armed = False
        self._nod_start = None
        self._nod_armed = False

    def _collect_eye(self, t, closed):
        if not closed:
            if self._eye_start is not None and t - self._eye_start >= self.cfg.long_closure_sec * 1000:
                self._long_events.append(t)
            self._eye_start = None
            self._eye_armed = True
        elif self._eye_armed and self._eye_start is None:
            self._eye_start = t

    def _collect_yawn(self, t, mar):
        if mar is None:
            self._mouth_start = self._mouth_low = None
            self._mouth_counted = self._mouth_armed = False
            return
        if mar < self.cfg.yawn_mar:
            if self._mouth_low is None:
                self._mouth_low = t
            if t - self._mouth_low >= self.cfg.mouth_close_grace_ms:
                self._mouth_start = None
                self._mouth_counted = False
                self._mouth_armed = True
        else:
            self._mouth_low = None
            if self._mouth_armed and self._mouth_start is None:
                self._mouth_start = t
            if (self._mouth_armed and not self._mouth_counted and self._mouth_start is not None
                    and t - self._mouth_start >= self.cfg.yawn_sec * 1000):
                self._yawn_events.append(t)
                self._mouth_counted = True

    def _collect_nod(self, ctx, t):
        gaze = ctx.results.get("gaze")
        info = gaze.info if gaze else {}
        value = gaze.value if gaze and isinstance(gaze.value, dict) else {}
        pitch, yaw = _number(value.get("pitch")), _number(value.get("yaw"))
        if (pause_reason(ctx) or not info.get("face") or "Calibrating" in info.get("message", "")
                or pitch is None or yaw is None or abs(yaw) > self.cfg.nod_yaw_max_deg):
            self._nod_start = None
            self._nod_armed = False
            return
        if abs(pitch) <= self.cfg.nod_forward_deg:
            if self._nod_start is not None:
                elapsed = (t - self._nod_start) / 1000.0
                if self.cfg.nod_min_sec <= elapsed <= self.cfg.nod_max_sec:
                    self._nod_events.append(t)
            self._nod_start = None
            self._nod_armed = True
        elif pitch >= self.cfg.nod_down_deg and self._nod_armed:
            self._nod_start = t
            self._nod_armed = False
        # 2초를 넘는 숙임은 끄덕임이 아니고 기존 미주시 분석기가 담당한다.
        if self._nod_start is not None and t - self._nod_start > self.cfg.nod_max_sec * 1000:
            self._nod_start = None

    def update(self, ctx):
        t = ctx.timestamp_ms
        c = self.cfg
        if self._last_t is not None and t <= self._last_t:
            # 타임스탬프 되감김/중복: 잘못된 사건/유지시간이 생기지 않게 새 관측으로 시작.
            self.reset()
        dt = t - self._last_t if self._last_t is not None else None
        gap = dt is not None and dt > c.max_gap_ms
        faces = getattr(ctx.face_result, "face_landmarks", []) or []
        eye = ctx.results.get("eye_state")
        ei = eye.info if eye else {}
        closed = ei.get("closed")
        valid = bool(len(faces) == 1 and ei.get("face") and ei.get("calibrated")
                     and closed is not None)
        if self._stream_start is None or gap:
            self._stream_start = t
        if gap:
            self._coverage.clear()
            self._high_since = self._low_since = None
            self._break_candidates()
        if dt is not None and 0 < dt <= c.max_gap_ms and valid and self._last_valid:
            self._coverage.append((self._last_t, t))
        cutoff = t - c.quality_window_sec * 1000
        while self._coverage and self._coverage[0][1] <= cutoff:
            self._coverage.popleft()
        valid_ms = sum(end - max(start, cutoff) for start, end in self._coverage)
        before_counts = (len(self._long_events), len(self._yawn_events), len(self._nod_events))
        if valid:
            self._collect_eye(t, bool(closed))
            mar = None
            if len(faces[0]) > 308:
                mar = _number(mouth_aspect_ratio(landmarks_to_pixels(faces[0], ctx.width, ctx.height)))
            self._collect_yawn(t, mar)
            self._collect_nod(ctx, t)
        else:
            mar = None
            self._break_candidates()
        new_events = tuple(len(events) > n for events, n in zip(
            (self._long_events, self._yawn_events, self._nod_events), before_counts))
        self._last_t, self._last_valid = t, valid
        for events, window in ((self._long_events, c.long_window_sec),
                               (self._yawn_events, c.yawn_window_sec), (self._nod_events, c.nod_window_sec)):
            while events and events[0] <= t - window * 1000:
                events.popleft()
        perclos = ctx.results.get("perclos")
        p = _number(perclos.value) if perclos else None
        p_valid = p is not None and 0 <= p <= 1
        ready = bool(valid and p_valid and t - self._stream_start >= c.quality_window_sec * 1000
                     and valid_ms >= c.quality_window_sec * 1000 * c.min_coverage)
        score, parts, multiplier = calculate_early_score(
            p, len(self._long_events), len(self._yawn_events), len(self._nod_events), c)
        activated = cleared = False
        if not ready:
            self._high_since = self._low_since = None
            # 이미 발생한 경고 이력은 유지. 관측 불가 상태는 정상 회복으로 해석하지 않는다.
            state = "unavailable" if not valid or not p_valid else "collecting"
        else:
            if not self._active:
                self._low_since = None
                if score >= c.threshold:
                    if self._high_since is None: self._high_since = t
                    if t - self._high_since >= c.confirm_sec * 1000:
                        self._active = True
                        activated = True
                else:
                    self._high_since = None
            else:
                self._high_since = None
                if score < c.clear_threshold:
                    if self._low_since is None: self._low_since = t
                    if t - self._low_since >= c.clear_sec * 1000:
                        self._active = False
                        self._low_since = None
                        cleared = True
                else:
                    self._low_since = None
            state = "warning" if self._active else "pending" if self._high_since is not None else "normal"
        reasons = [f"PERCLOS={p:.1%}" if p_valid else "PERCLOS=NA",
                   f"long={len(self._long_events)}/60s", f"yawn={len(self._yawn_events)}/180s",
                   f"nod={len(self._nod_events)}/60s", f"eye_multiplier={multiplier:g}"]
        return AnalyzerResult(self.name, score, "warning" if ready and self._active else "none", {
            "long_event": new_events[0], "yawn_event": new_events[1], "nod_event": new_events[2],
            "ready": ready, "state": state, "latched": self._active, "activated": activated,
            "cleared": cleared, "threshold": c.threshold, "clear_threshold": c.clear_threshold,
            "parts": parts, "eye_multiplier": multiplier, "perclos": p if p_valid else None,
            "long_count": len(self._long_events), "yawn_count": len(self._yawn_events),
            "nod_count": len(self._nod_events), "mar": mar,
            "valid_sec": valid_ms / 1000, "coverage": valid_ms / (c.quality_window_sec * 1000),
            "pending_sec": (t - self._high_since) / 1000 if self._high_since is not None else 0.0,
            "repeat_sec": c.repeat_sec, "reason": "; ".join(reasons),
        })
