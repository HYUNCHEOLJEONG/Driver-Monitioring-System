import argparse
import socket
import time
import cv2
from config import Config
from src.camera_capture import open_camera
from src.dashboard import Dashboard
from src.face_detector import FaceDetector
from src.object_detector import ObjectDetector
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="DMS Unity UDP & keyboard vehicle simulation")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--sim-speed-kmh", type=float, default=None)
    args = parser.parse_args()

    if args.list_cameras:
        for index in range(6):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            try:
                ok, frame = cap.read() if cap.isOpened() else (False, None)
                print(f"index {index}: {'OK ' + str(frame.shape) if ok else '-'}")
            finally:
                cap.release()
        return

    cfg = Config()
    if args.camera is not None:
        cfg.camera_index = args.camera

    face = FaceDetector(cfg.face_model, num_faces=2)
    pipeline = None
    dashboard = Dashboard()

    # 유니티(Unity)로 UDP 통신을 보낼 소켓 설정 (포트 8080)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_address = ("127.0.0.1", 8080)

    manual_speed_kph = None

    speed_presets = {
        ord("0"): 0.0,
        ord("1"): 30.0,
        ord("2"): 50.0,
        ord("3"): 80.0,
    }

    def set_vehicle_speed(speed_kmh):
        try:
            from src.warning_rules import valid_speed
            spd = valid_speed(speed_kmh)
        except Exception:
            spd = float(speed_kmh)

        vs = getattr(pipeline, "vehicle_source", None)
        if vs is not None:
            if hasattr(vs, "set_speed"):
                try:
                    vs.set_speed(spd)
                except Exception:
                    pass
            if hasattr(vs, "state") and hasattr(vs.state, "speed_kph"):
                vs.state.speed_kph = spd
            if hasattr(vs, "_speed_kph"):
                vs._speed_kph = spd
            if hasattr(vs, "speed_kph"):
                vs.speed_kph = spd
            store = getattr(vs, "store", None)
            if store is not None and hasattr(store, "put"):
                try:
                    store.put("speed_kph", spd)
                except Exception:
                    pass
        return spd

    try:
        obj = ObjectDetector(cfg.yolo_model, cfg.yolo_conf)
        pipeline = Pipeline(face, obj, cfg)

        if args.sim_speed_kmh is not None:
            manual_speed_kph = set_vehicle_speed(args.sim_speed_kmh)

        camera = open_camera(cfg)
        pipeline.events.emit(
            "SESSION",
            f"모니터링 시작 · 카메라={cfg.camera_index} · 모델={cfg.yolo_model.name}",
        )

        def get_analyzer(name):
            if hasattr(pipeline, "analyzers"):
                if isinstance(pipeline.analyzers, dict):
                    return pipeline.analyzers.get(name)
                elif isinstance(pipeline.analyzers, list):
                    for a in pipeline.analyzers:
                        if getattr(a, "name", None) == name:
                            return a
            return None

        window_name = "Driver Monitoring System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 900)

        try:
            start = previous = time.monotonic()
            last_ts = -1
            preview_shown = False
            failed_frames = 0

            while True:
                if (
                    preview_shown
                    and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                ok, frame = camera.read()
                if not ok or frame is None or not frame.size:
                    failed_frames += 1
                    if failed_frames == 1:
                        pipeline.events.emit(
                            "CAMERA", "프레임 수신 실패 · 재시도 중", "WARN"
                        )
                    if failed_frames >= 5:
                        print("Camera stopped delivering frames (5 consecutive failures).")
                        break
                    if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                    continue
                if failed_frames:
                    pipeline.events.emit("CAMERA", "프레임 수신 복구")
                failed_frames = 0

                if frame.ndim == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = ts

                ctx = pipeline.process(frame, ts, speed_kmh=manual_speed_kph)

                # 대시보드 내부 판정 결과 동기화
                dashboard.result = (
                    ctx.results.get("phone_state")
                    or ctx.results.get("phone")
                    or {"state": "no_phone"}
                )
                dashboard.smoking_result = (
                    ctx.results.get("smoking_state")
                    or ctx.results.get("smoking")
                    or {"state": "unknown"}
                )

                # =========================================================
                # [DMS 종합 위험도 + CAN 차량 속도 전송]
                # =========================================================
                speed_str = "N/A"
                gear_str = "-"
                try:
                    veh = getattr(ctx, "vehicle", None)
                    raw_spd = None
                    if veh is not None:
                        raw_spd = getattr(veh, "speed_kph", None)
                        if raw_spd is None:
                            raw_spd = getattr(veh, "speed", None)
                        raw_gear = getattr(veh, "gear", None)
                        if raw_gear:
                            gear_str = str(raw_gear)

                    if raw_spd is None and manual_speed_kph is not None:
                        raw_spd = manual_speed_kph

                    if raw_spd is not None:
                        speed_str = f"{float(raw_spd):.1f}"

                    gaze_analyzer = get_analyzer("gaze")
                    is_head_unstable = False

                    if gaze_analyzer:
                        state = getattr(gaze_analyzer, "current_unstable_state", "normal")
                        if state in ["alarm", "head_down"]:
                            is_head_unstable = True

                    phone_holding = (
                        dashboard.result.get("state") == "holding"
                        if isinstance(dashboard.result, dict)
                        else getattr(dashboard.result, "state", None) == "holding"
                    )
                    smoking_active = (
                        dashboard.smoking_result.get("state") == "smoking"
                        if isinstance(dashboard.smoking_result, dict)
                        else getattr(dashboard.smoking_result, "state", None) == "smoking"
                    )

                    if not phone_holding and hasattr(ctx, "objects") and ctx.objects:
                        for obj_item in ctx.objects:
                            label = str(obj_item.get("label", "")).lower()
                            if "phone" in label or label == "cell_phone":
                                phone_holding = True
                                break

                    if not smoking_active and hasattr(ctx, "objects") and ctx.objects:
                        for obj_item in ctx.objects:
                            label = str(obj_item.get("label", "")).lower()
                            if "cigarette" in label or "smoke" in label:
                                smoking_active = True
                                break

                    stage = getattr(dashboard, "current_stage", 0)
                    stage_danger = (stage >= 3)

                    if stage_danger:
                        level = "DANGER"
                    elif is_head_unstable or smoking_active or phone_holding:
                        level = "CAUTION"
                    else:
                        level = "NORMAL"

                    udp_payload = (
                        f"LEVEL={level}|STAGE={stage}|"
                        f"PHONE={phone_holding}|SMOKING={smoking_active}|GAZE={is_head_unstable}|"
                        f"SPEED={speed_str}|GEAR={gear_str}"
                    )
                    udp_sock.sendto(udp_payload.encode("utf-8"), unity_address)

                    print(
                        f"[실시간 DMS 연동] 송신: {level} | 속도:{speed_str}km/h ({gear_str}) "
                        f"(단계:{stage}, 폰:{phone_holding}, 흡연:{smoking_active}, 고개불안정:{is_head_unstable})"
                    )

                except Exception as e:
                    print(f"[UDP 전송 에러] {e}")
                # =========================================================

                now = time.monotonic()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now

                rendered_frame = dashboard.render(
                    ctx,
                    fps=fps,
                    camera_index=cfg.camera_index,
                    draw_landmarks=cfg.draw_landmarks,
                )

                cv2.imshow(window_name, rendered_frame)
                preview_shown = True

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in speed_presets:
                    manual_speed_kph = speed_presets[key]
                    applied_spd = set_vehicle_speed(manual_speed_kph)
                    pipeline.events.emit(
                        "VEHICLE", f"차량 속도 수동 변경: {applied_spd:.1f} km/h"
                    )
                    print(f"[차량 속도 변경] {applied_spd:.1f} km/h")
                elif key in (ord("r"), ord("R")):
                    gaze_analyzer = get_analyzer("gaze")
                    if gaze_analyzer is not None:
                        gaze_analyzer.reset()
                        pipeline.events.emit(
                            "GAZE", "전방 자세 재보정 시작 · 정면을 2초간 보세요"
                        )
                elif key in (ord("b"), ord("B")):
                    gaze_analyzer = get_analyzer("gaze")
                    if gaze_analyzer and hasattr(gaze_analyzer, "toggle_bypass"):
                        gaze_analyzer.toggle_bypass()
                elif hasattr(pipeline, "on_key") and pipeline.on_key(key):
                    pass
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        if pipeline is not None:
            pipeline.events.emit("SESSION", "모니터링 종료")
            pipeline.close()
        face.close()
        udp_sock.close()


if __name__ == "__main__":
    main()