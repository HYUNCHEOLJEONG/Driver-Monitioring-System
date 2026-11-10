"""차량(CAN) 연동 + 경고 예외 설정. config.py 와 분리되어 있어 단독으로 수정 가능."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class VehicleConfig:
    # ── 데이터 소스 ───────────────────────────────────────────
    # none   : 차량 데이터 없음 (예외 없음, 기존 동작과 동일)
    # manual : 키보드로 모의 입력  r=후진  a=좌측등  d=우측등  h=비상등
    # replay : data/sample_can.csv 재생 (CAN 하드웨어 없이 테스트)
    # can    : python-can 으로 실제 차량 버스 수신
    source: str = "manual"
    dbc_path: Path = ROOT / "data" / "sample.dbc"     # 실제 차량용 DBC 로 교체
    replay_path: Path = ROOT / "data" / "sample_can.csv"
    replay_loop: bool = True

    # ── 실제 CAN (source="can") ───────────────────────────────
    can_interface: str = "pcan"          # pcan | kvaser | vector | slcan | gs_usb | socketcan | virtual
    can_channel: str = "PCAN_USBBUS1"    # slcan 이면 "COM5" 등
    can_bitrate: int = 500000
    can_listen_only: bool = True         # 수신 전용 (버스에 아무것도 송신하지 않음)

    # DBC 신호 이름 매핑: 내부 필드 -> DBC 의 신호 이름 (차량마다 다름)
    signals: dict = field(default_factory=lambda: {
        "gear": "Gear",
        "turn_left": "TurnLeft",
        "turn_right": "TurnRight",
        "hazard": "Hazard",
        "speed_kph": "VehicleSpeed",
        "steering_deg": "SteeringAngle",
    })
    gear_map: dict = field(default_factory=lambda: {0: "P", 1: "R", 2: "N", 3: "D"})

    # ── 경고 예외 규칙 ────────────────────────────────────────
    stale_timeout_s: float = 1.0         # 이 시간 넘게 신호가 안 오면 예외 해제(안전 쪽으로 동작)
    exempt_analyzers: tuple = ("gaze",)  # 예외를 적용할 분석기 (졸음 계열은 적용되지 않음)

    reverse_exempt: bool = True          # 후진(R)일 때
    reverse_max_kph: float = 15.0        #   ...이 속도 이하에서만
    turn_exempt: bool = True             # 방향지시등 ON 일 때
    turn_window_sec: float = 8.0         #   ...켜진 뒤 이 시간 동안만 (차선 변경 확인용)
    turn_blink_hold_sec: float = 1.0     #   점멸(ON/OFF 반복)을 하나의 연속 ON 으로 묶는 시간
    hazard_exempt: bool = False          # 비상등은 기본적으로 예외 아님
    parked_exempt: bool = False          # 정차(P/N, 저속)
    parked_max_kph: float = 2.0