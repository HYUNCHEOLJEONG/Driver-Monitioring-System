import time
import winsound
import numpy as np
import cv2
from ._stub import StubAnalyzer
from ..types import FrameContext, AnalyzerResult
from ..geometry import landmarks_to_pixels, calculate_head_pose_angles

class GazeAnalyzer(StubAnalyzer):
    name = "gaze"
    requires = ()

    def __init__(self, yaw_threshold=30.0, pitch_threshold=18.0, calibration_sec=2.0, **kw):  # [수정] pitch_threshold를 22.0 -> 18.0으로 낮춰 숙임 감지를 더 민감하게 조정
        super().__init__(**kw)
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.calib_duration = calibration_sec
        
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.calib_samples_yaw = []
        self.calib_samples_pitch = []
        self.calib_start_time = None
        self.is_calibrated = False

        self.bypass_until = 0.0  
        # 버튼 좌표
        self.button_rect = (20, 160, 210, 190)

        self.state_start_time = None
        self.current_unstable_state = "normal"

    def reset(self):
        self.bypass_until = 0.0
        self.is_calibrated = False
        self.calib_start_time = None
        self.calib_samples_yaw.clear()
        self.calib_samples_pitch.clear()
        self.state_start_time = None
        self.current_unstable_state = "normal"

    def toggle_bypass(self):
        if time.time() < self.bypass_until:
            self.bypass_until = 0.0
            print("[INFO] Mute Mode Disabled.")
        else:
            self.bypass_until = time.time() + 180.0
            print("[INFO] 3-Min Mute Mode Activated!")

    def handle_mouse_click(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = self.button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.toggle_bypass()

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        current_time = time.time()
        is_bypassed = current_time < self.bypass_until

        faces = getattr(ctx.face_result, "face_landmarks", None)
        if not faces:
            self.state_start_time = None
            self.current_unstable_state = "normal"
            return AnalyzerResult(
                name=self.name,
                value=None,
                alert="none",
                info={"face": False, "state": "normal", "message": "Head Forward", "bypassed": is_bypassed, "duration": 0.0}
            )

        pts = landmarks_to_pixels(faces[0], ctx.width, ctx.height)
        raw_yaw, raw_pitch = calculate_head_pose_angles(pts)

        if not self.is_calibrated:
            if self.calib_start_time is None:
                self.calib_start_time = current_time
            
            self.calib_samples_yaw.append(raw_yaw)
            self.calib_samples_pitch.append(raw_pitch)

            if current_time - self.calib_start_time >= self.calib_duration:
                self.base_yaw = float(np.median(self.calib_samples_yaw))
                self.base_pitch = float(np.median(self.calib_samples_pitch))
                self.is_calibrated = True

            return AnalyzerResult(
                name=self.name,
                value={"yaw": 0.0, "pitch": 0.0},
                alert="none",
                info={"face": True, "state": "normal", "message": "Head Forward (Calibrating...)", "bypassed": is_bypassed, "duration": 0.0}
            )

        norm_yaw = raw_yaw - self.base_yaw
        norm_pitch = raw_pitch - self.base_pitch

        instant_state = "normal"
        if norm_yaw < -self.yaw_threshold:
            instant_state = "head_right"
        elif norm_yaw > self.yaw_threshold:
            instant_state = "head_left"
        elif norm_pitch < -self.pitch_threshold:
            instant_state = "head_up"
        elif norm_pitch > self.pitch_threshold:
            instant_state = "head_down"

        if instant_state == "normal":
            self.current_unstable_state = "normal"
            self.state_start_time = None
        else:
            if self.current_unstable_state != instant_state:
                self.current_unstable_state = instant_state
                self.state_start_time = current_time

        duration = 0.0
        if self.state_start_time is not None:
            duration = current_time - self.state_start_time

        state = "normal"
        msg = "Head Forward"
        alert_level = "none"

        if instant_state == "head_right":
            state = "head_right"
            if duration >= 3.5:
                msg = "Head RIGHT [WARNING]"
                alert_level = "warning"
            else:
                msg = f"Head RIGHT ({duration:.1f}s)"
        elif instant_state == "head_left":
            state = "head_left"
            if duration >= 3.5:
                msg = "Head LEFT [WARNING]"
                alert_level = "warning"
            else:
                msg = f"Head LEFT ({duration:.1f}s)"
        elif instant_state == "head_up":
            state = "head_up"
            if duration >= 3.5:
                msg = "Head UP [WARNING]"
                alert_level = "warning"
            else:
                msg = f"Head UP ({duration:.1f}s)"
        elif instant_state == "head_down":
            if is_bypassed:
                state = "normal"
                msg = "Head Forward (Muted)"
            else:
                state = "alarm"
                msg = "Head DOWN [ALARM]"
                alert_level = "alarm"
                try:
                    winsound.Beep(1200, 100)
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
                "bypass_left": max(0, int(self.bypass_until - current_time)),
                "duration": duration
            }
        )

    def draw(self, frame, result: AnalyzerResult):
        info = result.info
        if not info.get("face"):
            return frame

        state = info.get("state", "normal")
        msg = info.get("message", "Head Forward")
        is_bypassed = info.get("bypassed", False)
        duration = info.get("duration", 0.0)

        # UI 배경 대시보드 박스 좌표
        px1, py1, px2, py2 = 10, 90, 265, 198

        #반투명 효과
        overlay = frame.copy()

        #패널 배경 -----> 다크 네이버/차콜
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 20, 30), -1)

        #상태에 따른 패널 외곽 테두리 색상 변화
        if state == "normal":
            panel_border = (46, 204, 113)
            text_color = (80, 255, 130)
        elif state in ["head_up", "head_left", "head_right"]:
            panel_border = (0, 140, 255)
            text_color = (0, 165, 255)
        else:
            panel_border = (0, 0, 255)
            text_color = (0, 0, 255)

        #투명도 적용
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #패널 아웃라인 테두리 그림
        cv2.rectangle(frame, (px1, py1), (px2, py2), panel_border, 1)
        cv2.rectangle(frame, (px1, py1), (px2, py1 + 22), panel_border, -1)
        cv2.putText(frame, "AI GAZE MONITORING", (px1 + 15, py1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        #Gaze 상태 테스트 출력
        text = f"Gaze: {msg}"
        cv2.putText(frame, text, (px1 + 18, py1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)

        #3.5초 이상 지속 경고 --->깜박임 효과
        if state in ["head_up", "head_left", "head_right"]:
            bar_x1, bar_y1, bar_x2, bar_y2 = px1 + 18, py1 + 60, px2 - 18, py1 + 68
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            fill_width = int((min(duration, 3.5) / 3.5) * (bar_x2 - bar_x1))
            bar_color = (0, 165, 255) if duration < 3.5 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2), bar_color, -1)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (150, 150, 150), 1)

        if state in ["head_up", "head_left", "head_right"] and duration >= 3.5:
            if int(time.time() * 5) % 2 == 0:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 5)
        elif state == "alarm":
            if int(time.time() * 6) % 2 == 0:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        #Mute 버튼 렌더링
        bx1, by1, bx2, by2 = self.button_rect
        if is_bypassed:
            left_sec = info.get("bypass_left", 0)
            btn_text = f"MUTE ACTIVE ({left_sec}s)"
            bg_color = (25, 90, 25)
            btn_border = (40, 220, 40)
            btn_text_col = (255, 255, 255)
        else:
            btn_text = "[ 3m AUTO MUTE ]"
            bg_color = (40, 45, 55)
            btn_border = (120, 130, 150)
            btn_text_col = (220, 230, 240)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_border, 1)
        cv2.putText(frame, btn_text, (bx1 + 10, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, btn_text_col, 1, cv2.LINE_AA)

        return frame