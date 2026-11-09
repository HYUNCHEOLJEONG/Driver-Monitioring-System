import time
import threading
import numpy as np
import cv2
try:
    import winsound          # Windows 전용. 다른 OS 에서도 import 에러 없이 무음으로 동작
except ImportError:
    winsound = None
from ._stub import StubAnalyzer
from ..types import FrameContext, AnalyzerResult
from ..geometry import landmarks_to_pixels, calculate_head_pose_angles


class GazeAnalyzer(StubAnalyzer):
    """전방 미주시 감지 (팀 최신 버전 + CAN 경고 예외 병합).

    CAN 예외(ctx.exemptions['gaze'])가 있으면 좌/우/위 주시는 경고하지 않고 타이머도 리셋한다.
    고개 숙임(Head DOWN)은 졸음 신호이므로 예외를 적용하지 않는다.
    """
    name = "gaze"
    requires = ()

    def __init__(self, yaw_threshold=30.0, pitch_threshold=18.0, calibration_sec=2.0,
                 beep=True, beep_interval_s=0.5, **kw):
        super().__init__(**kw)
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.calib_duration = calibration_sec
        self.beep = beep
        self.beep_interval_s = beep_interval_s
        self._last_beep = 0.0

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

    def _beep(self, freq=1200, ms=100):
        """메인 루프를 막지 않도록 별도 스레드에서, 일정 간격으로만 울린다."""
        now = time.time()
        if not self.beep or winsound is None or now - self._last_beep < self.beep_interval_s:
            return
        self._last_beep = now
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()

    def update(self, ctx: FrameContext) -> AnalyzerResult:
        current_time = time.time()
        is_bypassed = current_time < self.bypass_until
        exempt = (getattr(ctx, "exemptions", None) or {}).get(self.name)   # CAN 예외 사유 or None

        faces = getattr(ctx.face_result, "face_landmarks", None)
        if not faces:
            self.state_start_time = None
            self.current_unstable_state = "normal"
            return AnalyzerResult(
                name=self.name, value=None, alert="none",
                info={"face": False, "state": "normal", "message": "Head Forward",
                      "bypassed": is_bypassed, "duration": 0.0,
                      "exempt": exempt, "self_exempt": True}
            )

        pts = landmarks_to_pixels(faces[0], ctx.width, ctx.height)
        raw_yaw, raw_pitch = calculate_head_pose_angles(pts, ctx.width, ctx.height)

        if not self.is_calibrated:
            if self.calib_start_time is None:
                self.calib_start_time = current_time

            if not exempt:   # 후진/방향지시등으로 옆·뒤를 보는 중의 값은 기준선에 섞지 않는다
                self.calib_samples_yaw.append(raw_yaw)
                self.calib_samples_pitch.append(raw_pitch)

            if current_time - self.calib_start_time >= self.calib_duration and self.calib_samples_yaw:
                self.base_yaw = float(np.median(self.calib_samples_yaw))
                self.base_pitch = float(np.median(self.calib_samples_pitch))
                self.is_calibrated = True

            return AnalyzerResult(
                name=self.name,
                value={"yaw": 0.0, "pitch": 0.0},
                alert="none",
                info={"face": True, "state": "normal", "message": "Head Forward (Calibrating...)",
                      "bypassed": is_bypassed, "duration": 0.0,
                      "exempt": exempt, "self_exempt": True}
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

        # CAN 예외: 좌/우/위 주시는 정상으로 간주하고 타이머를 리셋한다 (Head DOWN 은 제외)
        raw_state = None
        exempted = False
        if exempt and instant_state in ("head_left", "head_right", "head_up"):
            raw_state = instant_state
            instant_state = "normal"
            exempted = True

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

        if instant_state in ("head_right", "head_left", "head_up"):
            label = {"head_right": "RIGHT", "head_left": "LEFT", "head_up": "UP"}[instant_state]
            state = instant_state
            if duration >= 3.5:
                msg = f"Head {label} [WARNING]"
                alert_level = "warning"
            else:
                msg = f"Head {label} ({duration:.1f}s)"
        elif instant_state == "head_down":
            if is_bypassed:
                state = "normal"
                msg = "Head Forward (Muted)"
            else:
                state = "alarm"
                msg = "Head DOWN [ALARM]"
                alert_level = "alarm"
                self._beep(1200, 100)
        elif exempted:
            state = "exempt"
            msg = f"Head Forward (Exempt: {exempt})"

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
                "duration": duration,
                "exempt": exempt,
                "raw_state": raw_state,
                "self_exempt": True,   # 파이프라인이 alert 를 덮어쓰지 않도록 (예외는 여기서 직접 처리)
            }
        )

    def draw(self, frame, result: AnalyzerResult):
        info = result.info

        # 버튼은 얼굴이 안 잡혀도 항상 그린다
        is_bypassed = info.get("bypassed", False)

        if info.get("face"):
            state = info.get("state", "normal")
            msg = info.get("message", "Head Forward")
            duration = info.get("duration", 0.0)

            px1, py1, px2, py2 = 10, 90, 265, 198
            overlay = frame.copy()
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 20, 30), -1)

            if state == "normal":
                panel_border = (46, 204, 113)
                text_color = (80, 255, 130)
            elif state == "exempt":                      # CAN 예외 중
                panel_border = (200, 200, 0)
                text_color = (255, 255, 0)
            elif state in ["head_up", "head_left", "head_right"]:
                panel_border = (0, 140, 255)
                text_color = (0, 165, 255)
            else:
                panel_border = (0, 0, 255)
                text_color = (0, 0, 255)

            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.rectangle(frame, (px1, py1), (px2, py2), panel_border, 1)
            cv2.rectangle(frame, (px1, py1), (px2, py1 + 22), panel_border, -1)
            cv2.putText(frame, "AI GAZE MONITORING", (px1 + 15, py1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Gaze: {msg}", (px1 + 18, py1 + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)

            if state in ["head_up", "head_left", "head_right"]:
                bar_x1, bar_y1, bar_x2, bar_y2 = px1 + 18, py1 + 60, px2 - 18, py1 + 68
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
                fill_width = int((min(duration, 3.5) / 3.5) * (bar_x2 - bar_x1))
                bar_color = (0, 165, 255) if duration < 3.5 else (0, 0, 255)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2), bar_color, -1)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (150, 150, 150), 1)

            if state in ["head_up", "head_left", "head_right"] and duration >= 3.5:
                if int(time.time() * 5) % 2 == 0:
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 5)
            elif state == "alarm":
                if int(time.time() * 6) % 2 == 0:
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        # Mute 버튼
        bx1, by1, bx2, by2 = self.button_rect
        if is_bypassed:
            btn_text = f"MUTE ACTIVE ({info.get('bypass_left', 0)}s)"
            bg_color, btn_border, btn_text_col = (25, 90, 25), (40, 220, 40), (255, 255, 255)
        else:
            btn_text = "[ 3m AUTO MUTE ]"
            bg_color, btn_border, btn_text_col = (40, 45, 55), (120, 130, 150), (220, 230, 240)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_border, 1)
        cv2.putText(frame, btn_text, (bx1 + 10, by1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, btn_text_col, 1, cv2.LINE_AA)
        return frame
