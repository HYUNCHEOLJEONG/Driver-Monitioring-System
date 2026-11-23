"""초기 졸음 프로토타입 시험값. 임상/공인 졸음 판정 기준이 아니다.

기존 EAR 기반 눈감김 비율을 그대로 사용한다(엄밀한 P80 PERCLOS와 다름).
점수는 매 프레임 누적하지 않고 최근 사건에서 다시 계산한다.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class EarlyDrowsinessConfig:
    threshold: float = 10.0
    confirm_sec: float = 3.0
    clear_threshold: float = 6.0
    clear_sec: float = 30.0
    repeat_sec: float = 60.0
    quality_window_sec: float = 30.0
    min_coverage: float = 0.80
    max_gap_ms: int = 500
    long_closure_sec: float = 0.5
    long_window_sec: float = 60.0
    yawn_window_sec: float = 180.0
    yawn_mar: float = 0.60
    yawn_sec: float = 1.5
    mouth_close_grace_ms: int = 250
    nod_window_sec: float = 60.0
    nod_down_deg: float = 15.0
    nod_forward_deg: float = 5.0
    nod_yaw_max_deg: float = 15.0
    nod_min_sec: float = 0.3
    nod_max_sec: float = 2.0
    perclos_points: tuple = ((0.10, 2.0), (0.15, 4.0), (0.25, 6.0), (0.30, 8.0))
    long_points: tuple = ((1, 1.0), (2, 2.0), (3, 4.0), (5, 6.0))
    yawn_points: tuple = ((1, 1.0), (2, 3.0), (3, 4.0))
    nod_points: tuple = ((1, 1.0), (2, 3.0), (3, 4.0))
    # (PERCLOS 하한, 긴 감김 횟수 하한, 눈 지표 배율)
    interactions: tuple = ((0.15, 3, 1.5), (0.25, 5, 2.0))
