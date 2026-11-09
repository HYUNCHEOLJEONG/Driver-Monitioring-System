"""차량 상태 -> 경고 예외 판정.

원칙
  1) 신호가 오래 안 오면(stale) 예외를 주지 않는다 (fail-safe).
  2) 졸음 계열(eye_closure, perclos, microsleep, occlusion, distraction)은 어떤 설정으로도 예외 불가.
  3) 방향지시등은 '켜진 뒤 turn_window_sec 동안만' 예외 (계속 켜 둔 채 한눈파는 것 방지).
"""
import math

NEVER_EXEMPT = frozenset({"eye_closure", "perclos", "microsleep", "occlusion", "distraction"})


class _Turn:
    def __init__(self):
        self.last_on = None
        self.onset = None


class ExemptionPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        bad = set(cfg.exempt_analyzers) & NEVER_EXEMPT
        if bad:
            print(f"[policy] 예외 불가 분석기 무시: {sorted(bad)}")
        self.targets = tuple(n for n in cfg.exempt_analyzers if n not in NEVER_EXEMPT)
        self._turn = {"left": _Turn(), "right": _Turn()}
        self._haz_last = None

    def _track(self, side, on, t):
        tr = self._turn[side]
        if on:
            if tr.last_on is None or t - tr.last_on > self.cfg.turn_blink_hold_sec:
                tr.onset = t                      # 점멸 사이 공백이 길면 새 신호로 간주
            tr.last_on = t
        active = tr.last_on is not None and t - tr.last_on <= self.cfg.turn_blink_hold_sec
        return active, (t - tr.onset) if active else 0.0

    def reasons(self, s, t_ms):
        c, t, T = self.cfg, t_ms / 1000.0, self.cfg.stale_timeout_s
        out = []

        # 비상등: 점멸하므로 hold 시간으로 묶는다. 비상등 중에는 좌/우 비트도 같이 켜지므로
        # 방향지시등 추적 자체를 건너뛴다 (비상등이 끝난 직후 가짜 '방향지시등'이 남는 것 방지).
        if s.fresh("hazard", T) and s.hazard:
            self._haz_last = t
        hazard = self._haz_last is not None and t - self._haz_last <= c.turn_blink_hold_sec
        if hazard:
            for tr in self._turn.values():
                tr.last_on = tr.onset = None
        for side, fld in (("left", "turn_left"), ("right", "turn_right")):
            if not hazard and s.fresh(fld, T):
                active, el = self._track(side, bool(getattr(s, fld)), t)
                if c.turn_exempt and active and not hazard and el <= c.turn_window_sec:
                    out.append(f"turn_{side} {el:.1f}s")

        speed_ok = lambda lim: (not s.fresh("speed_kph", T)) or abs(s.speed_kph) <= lim
        if c.reverse_exempt and s.fresh("gear", T) and s.gear == "R" and speed_ok(c.reverse_max_kph):
            out.append("reverse")
        if c.hazard_exempt and hazard:
            out.append("hazard")
        if (c.parked_exempt and s.fresh("gear", T) and s.gear in ("P", "N")
                and s.fresh("speed_kph", T) and abs(s.speed_kph) <= c.parked_max_kph):
            out.append("parked")
        return out

    def evaluate(self, state, t_ms):
        """-> {분석기 이름: 사유}. 비어 있으면 예외 없음."""
        r = self.reasons(state, t_ms)
        return {n: ", ".join(r) for n in self.targets} if r else {}


def draw_hud(frame, state, exemptions):
    import cv2
    h = frame.shape[0]
    if state.source == "none":
        txt, col = ("CAN: off" + (f" ({state.error[:40]})" if state.error else "")), (160, 160, 160)
    else:
        parts = [f"gear={state.gear or '-'}",
                 "L" if state.turn_left else "", "R" if state.turn_right else "",
                 "HAZ" if state.hazard else "",
                 f"{state.speed_kph:.0f}km/h" if state.speed_kph is not None else ""]
        txt, col = f"{state.source}: " + " ".join(p for p in parts if p), (255, 255, 0)
    cv2.putText(frame, txt, (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    if exemptions:
        names = ",".join(exemptions)
        why = next(iter(exemptions.values()))
        cv2.putText(frame, f"EXEMPT[{names}]: {why}", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
