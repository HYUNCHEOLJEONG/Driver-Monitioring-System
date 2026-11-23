"""화면·분석기에서 공유하는 차속별 경고 기준 (km/h, 초)."""
import math

def valid_speed(value):
    try:
        speed = float(value)
        return speed if math.isfinite(speed) and speed >= 0 else None
    except (TypeError, ValueError):
        return None

def gaze_limit(speed):
    speed = valid_speed(speed)
    if speed is None: return 3.5
    if speed < 20: return None
    if speed < 50: return 6.0
    if speed < 80: return 3.5
    return 2.0

def closure_limit(speed):
    speed = valid_speed(speed)
    return 1.0 if speed is not None and speed >= 80 else 2.0

def pause_reason(ctx):
    v = ctx.vehicle
    reasons = []
    if v and v.gear == "R": reasons.append("후진")
    if v and v.turn_left: reasons.append("좌측 방향지시등")
    if v and v.turn_right: reasons.append("우측 방향지시등")
    return " / ".join(reasons)