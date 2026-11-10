import os
from .types import FrameContext
from .event_logger import EventLogger
from .phone_presence import PhonePresence
from .dashboard import Dashboard
from .smoking_relation import SmokingRelation
from .analyzers import REGISTRY
from .vehicle import create_source
try:
    from vehicle_config import VehicleConfig
except ImportError:
    VehicleConfig = None


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

        # ── [핵심 수정] sample_can.csv와 sample.dbc의 절대/상대 경로 명시 및 Replay 강제 연동 ──
        base_dir = os.path.dirname(os.path.dirname(__file__))
        dbc_path = os.path.join(base_dir, "data", "sample.dbc")
        replay_path = os.path.join(base_dir, "data", "sample_can.csv")

        if VehicleConfig is not None:
            vehicle_cfg = VehicleConfig(source="replay", dbc_path=dbc_path, replay_path=replay_path, replay_loop=True)
        else:
            class DummyVehicleCfg:
                source = "replay"
                dbc_path = dbc_path
                replay_path = replay_path
                replay_loop = True
            vehicle_cfg = DummyVehicleCfg()

        self.vehicle_source = create_source(vehicle_cfg)
        self.vehicle_source.start()

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

    def process(self, frame, timestamp_ms):
        h, w = frame.shape[:2]
        ctx = FrameContext(frame=frame, timestamp_ms=timestamp_ms, width=w, height=h)
        
        # ── 매 프레임마다 재생 중인 CAN 데이터의 상태(속도 등)를 가져와서 주입 ──
        ctx.vehicle = self.vehicle_source.get_state()

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
            
        self.events.update(ctx, self.yolo_status.result, self.yolo_status.smoking_result)
        return ctx

    def close(self):
        if hasattr(self, "vehicle_source") and self.vehicle_source:
            self.vehicle_source.stop()