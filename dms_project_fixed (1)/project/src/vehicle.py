"""키보드 차량 입력. 샘플 CSV/DBC 및 실제 CAN 버스를 열지 않는다."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VehicleState:
    gear: Optional[str] = "D"
    turn_left: bool = False
    turn_right: bool = False
    hazard: bool = False
    speed_kph: Optional[float] = None
    steering_deg: Optional[float] = None
    ages: dict = field(default_factory=dict)
    source: str = "manual"
    error: str = ""

    @property
    def speed(self):
        return self.speed_kph

    def fresh(self, name, timeout):
        return self.ages.get(name, float("inf")) <= timeout

class ManualSource:
    """0/1/2/3/C 차속, G/D 후진/주행, J/L 좌/우 토글. 값은 해제까지 유지."""
    def __init__(self):
        self.state = VehicleState()

    def start(self): pass
    def stop(self): pass

    def on_key(self, key):
        k = chr(key).lower() if 0 <= key <= 255 else ""
        if k in "0123" and k:
            self.state.speed_kph = {"0": 0.0, "1": 30.0, "2": 50.0, "3": 80.0}[k]
        elif k == "c": self.state.speed_kph = None
        elif k == "g": self.state.gear = "R"
        elif k == "d": self.state.gear = "D"
        elif k == "j": self.state.turn_left = not self.state.turn_left
        elif k == "l": self.state.turn_right = not self.state.turn_right
        else: return False
        return True

    def get_state(self):
        from dataclasses import replace
        return replace(self.state, ages={k: 0.0 for k in ("gear", "turn_left", "turn_right", "speed_kph")})
