from ._stub import StubAnalyzer
from ..types import FrameContext, AnalyzerResult

class FatigueAnalyzer(StubAnalyzer):
    """눈 감김 지속 시간 및 차량 CAN 데이터(주행 속도 등)를 복합적으로 반영하여
    주행 중일 때 피로도가 가중되도록 설계된 고도화된 피로도 분석기
    """
    name = "fatigue"
    requires = ("eye_closure", "perclos", "gaze")

    def __init__(self, fatigue_threshold=100.0, **kw):
        super().__init__(**kw)
        self.fatigue_score = 0.0
        self.threshold = fatigue_threshold
        self.rest_recommended = False

    def reset(self):
        self.fatigue_score = 0.0
        self.rest_recommended = False

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        res = ctx.results
        score_increment = 0.0

        # 1. 눈 감김 지속 시간(초) 가져오기
        closure = res.get("eye_closure")
        closure_sec = closure.value if closure and closure.value else 0.0

        # 2. CAN 데이터에서 차량 속도(Vehicle Speed) 등 주행 상태 확인
        # (ctx 내에 vehicle 또는 can 관련 정보가 담겨있거나 다루는 속성 활용)
        vehicle_info = getattr(ctx, "vehicle", None) or res.get("vehicle")
        speed = 0.0
        if vehicle_info:
            # vehicle_info가 딕셔너리이거나 객체일 경우 속도 추출
            speed = vehicle_info.get("speed", 0.0) if isinstance(vehicle_info, dict) else getattr(vehicle_info, "speed", 0.0)

        # 주행 속도가 일정 이상(예: 10km/h 이상)일 때는 위험 상황 가중치를 높이고, 정차 중이면 낮춤
        speed_multiplier = 1.5 if speed > 10.0 else 0.5

        # 기본 눈 감김 가중치에 CAN 속도 배율 적용
        if closure_sec >= 2.0:
            score_increment += 0.5 * speed_multiplier
        elif closure_sec >= 1.0:
            score_increment += 0.2 * speed_multiplier

        # PERCLOS 보조 가산
        perclos = res.get("perclos")
        if perclos and perclos.value and perclos.value > 0.3:
            score_increment += 0.2 * speed_multiplier

        # 안전 상태(눈을 뜨고 주행/대기)일 때는 점수 회복
        if score_increment == 0.0 and closure_sec == 0.0:
            self.fatigue_score = max(0.0, self.fatigue_score - 1.0)
        else:
            self.fatigue_score = max(0.0, min(100.0, self.fatigue_score + score_increment))
        
        # 휴식 권고 임계치 판정
        if self.fatigue_score >= self.threshold:
            self.rest_recommended = True
            alert_state = "alarm"
        elif self.fatigue_score < self.threshold * 0.4:
            self.rest_recommended = False
            alert_state = "none"
        elif self.fatigue_score >= self.threshold * 0.6:
            alert_state = "warning"
        else:
            alert_state = "none"

        return AnalyzerResult(
            name=self.name,
            value=round(self.fatigue_score, 1),
            alert=alert_state,
            info={
                "fatigue_score": round(self.fatigue_score, 1),
                "rest_recommended": self.rest_recommended,
                "admin_log_summary": f"누적 피로도: {self.fatigue_score:.1f}점 (속도: {speed:.1f}km/h)"
            }
        )