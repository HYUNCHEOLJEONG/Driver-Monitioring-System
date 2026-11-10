"""차량 신호 수신. 분석기들은 이 모듈을 직접 쓰지 않고 ctx.vehicle(VehicleState) 만 본다.

소스 종류: NullSource / ManualSource(키보드) / ReplaySource(CSV 재생) / CanSource(python-can)
"""
import csv
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

FIELDS = ("gear", "turn_left", "turn_right", "hazard", "speed_kph", "steering_deg")


@dataclass
class VehicleState:
    gear: Optional[str] = None            # P / R / N / D
    turn_left: Optional[bool] = None
    turn_right: Optional[bool] = None
    hazard: Optional[bool] = None
    speed_kph: Optional[float] = None
    steering_deg: Optional[float] = None
    ages: dict = field(default_factory=dict)   # 필드 -> 마지막 수신 후 경과(초)
    source: str = "none"
    error: str = ""

    def fresh(self, name: str, timeout: float) -> bool:
        """해당 신호가 timeout 안에 갱신됐는지. 한 번도 안 왔으면 False."""
        return self.ages.get(name, math.inf) <= timeout


class VehicleSource:
    name = "base"
    error = ""

    def start(self):
        pass

    def stop(self):
        pass

    def get_state(self) -> VehicleState:
        raise NotImplementedError

    def on_key(self, key: int) -> bool:
        return False


class _Store:
    """스레드 안전한 최신값 저장소."""

    def __init__(self, clock):
        self.clock = clock
        self._lock = threading.Lock()
        self._v = {}   # field -> (value, t)

    def put(self, name, value, t=None):
        with self._lock:
            self._v[name] = (value, self.clock() if t is None else t)

    def snapshot(self, source, error="") -> VehicleState:
        now = self.clock()
        with self._lock:
            items = dict(self._v)
        st = VehicleState(source=source, error=error)
        for k, (val, t) in items.items():
            setattr(st, k, val)
            st.ages[k] = max(0.0, now - t)
        return st


class NullSource(VehicleSource):
    name = "none"

    def __init__(self, error=""):
        self.error = error

    def get_state(self):
        return VehicleState(source="none", error=self.error)


class ManualSource(VehicleSource):
    """개발/시연용. r=후진 a=좌 d=우 h=비상등 (다시 누르면 해제)."""
    name = "manual"

    def __init__(self, cfg, clock=time.monotonic):
        self.clock = clock
        self._s = {"reverse": False, "left": False, "right": False, "hazard": False}
        self._last = {}

    def on_key(self, key):
        now = self.clock()
        if now - self._last.get(key, -9) < 0.4:   # 꾹 누름(키 반복)으로 토글이 깜박이는 것 방지
            return False
        self._last[key] = now
        m = {ord("r"): "reverse", ord("a"): "left", ord("d"): "right", ord("h"): "hazard"}
        if key in m:
            self._s[m[key]] = not self._s[m[key]]
            return True
        return False

    def get_state(self):
        s = self._s
        st = VehicleState(
            gear="R" if s["reverse"] else "D", turn_left=s["left"], turn_right=s["right"],
            hazard=s["hazard"], speed_kph=5.0 if s["reverse"] else 40.0, steering_deg=0.0,
            source="manual")
        st.ages = {k: 0.0 for k in FIELDS}
        return st


class _DbcSource(VehicleSource):
    """프레임(id, data) -> DBC 디코드 -> 내부 필드로 변환."""

    def __init__(self, cfg, clock=time.monotonic):
        import cantools  # 지연 import: 이 소스를 쓸 때만 필요
        self.cfg = cfg
        self.clock = clock
        self.db = cantools.database.load_file(str(cfg.dbc_path))
        self.store = _Store(clock)

    def _handle(self, can_id, data, t=None):
        try:
            sig = self.db.decode_message(can_id, bytes(data), decode_choices=False, scaling=True)
        except Exception:      # DBC 에 없는 ID / 길이 불일치는 무시
            return
        for fld, sname in self.cfg.signals.items():
            if sname not in sig:
                continue
            v = sig[sname]
            if fld == "gear":
                v = self.cfg.gear_map.get(int(v), "?")
            elif fld in ("turn_left", "turn_right", "hazard"):
                v = bool(int(v))
            else:
                v = float(v)
            self.store.put(fld, v, t)

    def get_state(self):
        return self.store.snapshot(self.name, self.error)


class ReplaySource(_DbcSource):
    """data/sample_can.csv (time_s,can_id,data_hex) 를 실시간 속도로 재생."""
    name = "replay"

    def __init__(self, cfg, clock=time.monotonic):
        super().__init__(cfg, clock)
        self.loop = cfg.replay_loop
        self.frames = []
        with open(cfg.replay_path, newline="") as f:
            for row in csv.DictReader(f):
                self.frames.append((float(row["time_s"]), int(row["can_id"], 16),
                                    bytes.fromhex(row["data"])))
        self.frames.sort(key=lambda x: x[0])
        self.duration = (self.frames[-1][0] + 0.1) if self.frames else 0.0
        self._t0 = None
        self._idx = 0
        self._last_el = -1.0

    def _advance(self):
        now = self.clock()
        if self._t0 is None:
            self._t0 = now
        el = now - self._t0
        if self.loop and self.duration > 0:
            el %= self.duration
        if el < self._last_el:           # 루프 되감김
            self._idx = 0
        self._last_el = el
        while self._idx < len(self.frames) and self.frames[self._idx][0] <= el:
            ft, cid, data = self.frames[self._idx]
            self._handle(cid, data, t=now - (el - ft))
            self._idx += 1

    def get_state(self):
        self._advance()
        return super().get_state()


class CanSource(_DbcSource):
    """python-can 백그라운드 수신. 송신은 절대 하지 않는다."""
    name = "can"

    def __init__(self, cfg, clock=time.monotonic):
        super().__init__(cfg, clock)
        self._bus = None
        self._run = False
        self._th = None

    def start(self):
        import can
        kw = dict(interface=self.cfg.can_interface, channel=self.cfg.can_channel)
        if self.cfg.can_interface != "virtual":
            kw["bitrate"] = self.cfg.can_bitrate
        if self.cfg.can_listen_only and self.cfg.can_interface in ("pcan", "kvaser", "vector"):
            kw["state"] = can.BusState.PASSIVE
        self._bus = can.Bus(**kw)
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while self._run:
            try:
                msg = self._bus.recv(timeout=0.1)
            except Exception as e:
                self.error = f"CAN recv error: {e}"
                time.sleep(0.5)
                continue
            if msg is not None and not msg.is_error_frame:
                self._handle(msg.arbitration_id, msg.data)

    def stop(self):
        self._run = False
        if self._th:
            self._th.join(timeout=1.0)
        if self._bus:
            self._bus.shutdown()


def create_source(cfg, clock=time.monotonic) -> VehicleSource:
    """설정으로 소스 생성. 실패하면 NullSource(예외 없음 = 안전 쪽)로 대체하고 사유를 남긴다."""
    kind = cfg.source
    try:
        if kind == "none":
            return NullSource()
        if kind == "manual":
            return ManualSource(cfg, clock)
        if kind == "replay":
            return ReplaySource(cfg, clock)
        if kind == "can":
            return CanSource(cfg, clock)
        raise ValueError(f"unknown vehicle source '{kind}'")
    except Exception as e:
        print(f"[vehicle] source '{kind}' unavailable -> exemptions disabled: {e}")
        return NullSource(error=str(e))