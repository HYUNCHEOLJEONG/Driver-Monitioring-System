import time
import winsound
import numpy as np
import cv2
from ._stub import StubAnalyzer
from ..types import FrameContext, AnalyzerResult
from ..geometry import landmarks_to_pixels, calculate_head_pose_angles

class GazeAnalyzer(StubAnalyzer):
    """전방 미주시 감지: 카메라 기준 자동 보정, 좌우 반전 교정, Gaze 하단 디자인 바이패스 버튼 적용."""
    name = "gaze"
    requires = ()

    def __init__(self, yaw_threshold=30.0, pitch_threshold=22.0, calibration_sec=2.0, **kw):
        super().__init__(**kw)
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.calib_duration = calibration_sec
        
        # 카메라 정면 자동 기준점(Calibration Offset) 변수
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.calib_samples_yaw = []
        self.calib_samples_pitch = []
        self.calib_start_time = None
        self.is_calibrated = False

        # 3분 바이패스 관련
        self.bypass_until = 0.0  
        
        # 미니 버튼 배치 (x1, y1, x2, y2)
        self.button_rect = (10, 120, 150, 148)

    def reset(self):
        """리셋 호출 시 바이패스 및 정면 기준선 재설정"""
        self.bypass_until = 0.0
        self.is_calibrated = False
        self.calib_start_time = None
        self.calib_samples_yaw.clear()
        self.calib_samples_pitch.clear()

    def toggle_bypass(self):
        """3분 바이패스 토글"""
        if time.time() < self.bypass_until:
            self.bypass_until = 0.0
            print("[INFO] Mute Mode Disabled.")
        else:
            self.bypass_until = time.time() + 180.0
            print("[INFO] 3-Min Mute Mode Activated!")

    def handle_mouse_click(self, event, x, y):
        """버튼 영역 마우스 클릭 감지"""
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = self.button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.toggle_bypass()

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        current_time = time.time()
        is_bypassed = current_time < self.bypass_until

        faces = getattr(ctx.face_result, "face_landmarks", None)
        if not faces:
            return AnalyzerResult(
                name=self.name,
                value=None,
                alert="none",
                info={"face": False, "state": "normal", "message": "Head Forward", "bypassed": is_bypassed}
            )

        pts = landmarks_to_pixels(faces[0], ctx.width, ctx.height)
        raw_yaw, raw_pitch = calculate_head_pose_angles(pts)

        # 카메라 위치 기준 자동 정면 칼리브레이션 (첫 2초간 정면을 바라보는 기준 수집)
        if not self.is_calibrated:
            if self.calib_start_time is None:
                self.calib_start_time = current_time
            
            self.calib_samples_yaw.append(raw_yaw)
            self.calib_samples_pitch.append(raw_pitch)

            if current_time - self.calib_start_time >= self.calib_duration:
                self.base_yaw = float(np.median(self.calib_samples_yaw))
                self.base_pitch = float(np.median(self.calib_samples_pitch))
                self.is_calibrated = True

            # 칼리브레이션 진행 중에는 무조건 흰색 Head Forward 상태 유지
            return AnalyzerResult(
                name=self.name,
                value={"yaw": 0.0, "pitch": 0.0},
                alert="none",
                info={"face": True, "state": "normal", "message": "Head Forward (Calibrating...)", "bypassed": is_bypassed}
            )

        # 기준 오프셋을 차감하여 현재 사용자가 보는 '카메라 정면'을 0.0으로 만듦
        norm_yaw = raw_yaw - self.base_yaw
        norm_pitch = raw_pitch - self.base_pitch

        state = "normal"
        msg = "Head Forward"
        alert_level = "none"

        # 좌/우 거울모드 반전 교정 및 상/하 각도 판정
        # 카메라 시점 기준: 고개를 오른쪽으로 돌리면 yaw가 음수, 왼쪽이면 양수
        if norm_yaw < -self.yaw_threshold:       # 오른쪽으로 고개를 돌릴 때
            state = "head_right"
            msg = "Head RIGHT [WARNING]"
            alert_level = "warning"
        elif norm_yaw > self.yaw_threshold:      # 왼쪽으로 고개를 돌릴 때
            state = "head_left"
            msg = "Head LEFT [WARNING]"
            alert_level = "warning"
        elif norm_pitch < -self.pitch_threshold: # 위쪽을 볼 때 (Head UP)
            state = "head_up"
            msg = "Head UP [WARNING]"
            alert_level = "warning"
        elif norm_pitch > self.pitch_threshold:  # 아래쪽으로 고개를 숙일 때 (Head DOWN)
            if is_bypassed:
                state = "normal"
                msg = "Head Forward (Muted)"
            else:
                state = "alarm"
                msg = "Head DOWN [ALARM]"
                alert_level = "alarm"
                try:
                    winsound.Beep(1200, 100) # 경고음
                except Exception:
                    pass
        else:
            state = "normal"
            msg = "Head Forward"

        return AnalyzerResult(
            name=self.name,
            value={"yaw": norm_yaw, "pitch": norm_pitch},
            alert=alert_level,
            info={
                "face": True,
                "state": state,
                "yaw": norm_yaw,
                "pitch": norm_pitch,
                "message": msg,
                "bypassed": is_bypassed,
                "bypass_left": max(0, int(self.bypass_until - current_time))
            }
        )

    def draw(self, frame, result: AnalyzerResult):
        info = result.info
        if not info.get("face"):
            return frame

        state = info.get("state", "normal")
        msg = info.get("message", "Head Forward")
        is_bypassed = info.get("bypassed", False)

        #  상태별 텍스트 색상 매핑
        if state == "normal":
            color = (255, 255, 255)  # 정면/초기: 흰색
        elif state == "head_up":
            color = (0, 255, 255)    # 위쪽: 노란색
        elif state in ["head_left", "head_right"]:
            color = (0, 165, 255)    # 좌/우: 주황색
        else:  
            color = (0, 0, 255)      # 숙임: 빨간색 + 알람

        text = f"Gaze: {msg}"

        # 1. 메인 Gaze 상태 텍스트 출력 (Y=100)
        cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. 미니 UI 버튼 그리기
        bx1, by1, bx2, by2 = self.button_rect
        
        if is_bypassed:
            left_sec = info.get("bypass_left", 0)
            btn_text = f"MUTE ({left_sec}s)"
            bg_color = (35, 120, 35)      # 세련된 다크 세이지 그린
            border_color = (50, 220, 50)  # 밝은 녹색 테두리
            text_color = (255, 255, 255)
        else:
            btn_text = "3m AUTO MUTE"
            bg_color = (45, 45, 45)       # 모던 패널 다크 그레이
            border_color = (120, 120, 120)# 은색 테두리
            text_color = (220, 220, 220)

        # 버튼 배경 및 패널 테두리
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, 1)

        # 버튼 내부 텍스트 
        cv2.putText(frame, btn_text, (bx1 + 10, by1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

        return frame