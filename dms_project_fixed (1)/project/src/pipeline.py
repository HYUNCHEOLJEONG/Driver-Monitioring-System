from .types import FrameContext
from .event_logger import EventLogger
from .phone_presence import PhonePresence
from .dashboard import Dashboard
from .smoking_relation import SmokingRelation
from .analyzers import REGISTRY
from .vehicle import ManualSource
from .warning_rules import valid_speed
from .warning_sound import WarningSound
from .early_drowsiness_output import EarlyDrowsinessOutput


class Pipeline:
    """검출기 실행 -> 분석기 의존 순서대로 실행 (CAN 차량 데이터 연동 포함)."""

    def __init__(self, face_detector, object_detector, config):
        self.face = face_detector
        self.obj = object_detector
        self.cfg = config
        self.events = EventLogger()
        self.analyzers = self._build(config)
        self._frame_idx = 0
        self._last_objects = []
        self.yolo_status = Dashboard()
        self.smoking_relation = SmokingRelation(
            hold_ms=config.smoking_hold_ms, min_samples=config.smoking_min_samples,
            max_gap_ms=config.object_max_gap_ms, roi=config.driver_roi,
            mouth_distance=config.smoking_mouth_distance)
        self.phone_relation = PhonePresence()

        self.vehicle_source = ManualSource()
        self.vehicle_source.start()
        self.warning_sound = WarningSound()
        self.early_output = EarlyDrowsinessOutput()

    def on_key(self, key):
        return self.vehicle_source.on_key(key)

    @staticmethod
    def _build(config):
        names = list(config.enabled_analyzers)
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise KeyError(f"Unknown analyzers: {unknown}. Available: {list(REGISTRY)}")
        ordered, seen = [], set()

        def visit(n):  # 의존성 위상 정렬
            if n in seen:
                return
            seen.add(n)
            for r in REGISTRY[n].requires:
                visit(r)
            ordered.append(n)

        for n in names:
            visit(n)
        return [REGISTRY[n](**config.analyzer_params.get(n, {})) for n in ordered]

    def process(self, frame, timestamp_ms, speed_kmh=None, gear=None, turn_signal=None):
        h, w = frame.shape[:2]
        ctx = FrameContext(frame=frame, timestamp_ms=timestamp_ms, width=w, height=h)
        
        # ── 매 프레임마다 재생 중인 CAN 데이터의 상태(속도 등)를 가져와서 주입 ──
        ctx.vehicle = self.vehicle_source.get_state()
        # 키보드 입력과 동일한 구조로 외부 모의 입력도 주입 가능.
        if speed_kmh is not None: ctx.vehicle.speed_kph = valid_speed(speed_kmh)
        if gear is not None: ctx.vehicle.gear = gear
        if turn_signal is not None:
            ctx.vehicle.turn_left = turn_signal in ("left", "both", "hazard")
            ctx.vehicle.turn_right = turn_signal in ("right", "both", "hazard")
        ctx.speed_kmh = valid_speed(ctx.vehicle.speed_kph)
        ctx.vehicle.speed_kph = ctx.speed_kmh
        ctx.gear = ctx.vehicle.gear
        ctx.turn_signal = ("both" if ctx.vehicle.turn_left and ctx.vehicle.turn_right else
                           "left" if ctx.vehicle.turn_left else "right" if ctx.vehicle.turn_right else "off")

        ctx.face_result = self.face.detect(frame, timestamp_ms)
        if self._frame_idx % max(1, self.cfg.yolo_every_n_frames) == 0:
            self._last_objects = self.obj.detect(frame)
            self.yolo_status.result = self.phone_relation.update(
                [o for o in self._last_objects if o['label'] == 'cell_phone'],
                ctx.face_result, w, h, timestamp_ms)
            self.yolo_status.smoking_result = self.smoking_relation.update(
                self._last_objects, ctx.face_result, w, h, timestamp_ms)
        ctx.objects = self._last_objects
        self._frame_idx += 1
        
        for a in self.analyzers:
            ctx.results[a.name] = a.update(ctx)
            
        self.warning_sound.update(ctx)
        self.events.update(ctx, self.yolo_status.result, self.yolo_status.smoking_result)
        self.early_output.update(ctx, self.events)
        return ctx

    def close(self):
        self.warning_sound.close()
        if hasattr(self, "vehicle_source") and self.vehicle_source:
            self.vehicle_source.stop()