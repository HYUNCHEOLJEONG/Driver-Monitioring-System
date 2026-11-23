# Driver Monitoring System - Detection Prototype

Uses OpenCV webcam capture, MediaPipe Face Landmarker (VIDEO mode), and Ultralytics YOLO11n.

## Setup

Use a compatible Python environment (Python 3.10 or 3.11 recommended).

```powershell
python -m pip install -r requirements.txt
```

Download and save model files as:

- `models/face_landmarker.task` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- `models/yolo11n.pt` from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt (or use Ultralytics' official auto-download via `YOLO('yolo11n.pt')` and move it into models/).

## Run

```powershell
python main.py
```

Press `q` to quit. Change `CAMERA_INDEX` in `main.py` when connecting a USB/NIR camera.

The model weights are not bundled in this source archive. The model output and FPS depend on hardware. YOLO11n pretrained COCO weights recognize cell phones but do not identify smoking as a dedicated class.

## 주행 리포트 메일 전송

종료(`q`/창 닫기) 시 하품·PERCLOS·휴대폰·흡연 보고서를 `data/reports/`에 HTML로 저장하고,
같은 보고서를 메일 본문 + 첨부로 보냅니다. 환경변수가 없으면 메일은 건너뛰고 내용만 터미널에 출력합니다.
추가 설치는 필요 없습니다 (Python 표준 라이브러리 smtplib).

Gmail: Google 계정에서 2단계 인증을 켠 뒤 보안 → 앱 비밀번호(16자리)를 만들어 사용합니다.

```powershell
$env:DMS_SMTP_USER = "보내는주소@gmail.com"
$env:DMS_SMTP_PASSWORD = "앱비밀번호16자리"
$env:DMS_MAIL_TO = "받는주소@gmail.com"     # 쉼표로 여러 명 가능
python main.py
```

- 네이버 메일: `$env:DMS_SMTP_HOST = "smtp.naver.com"` 추가 (메일 환경설정에서 SMTP 사용 켜기).
- 차속 10km/h 미만은 집계에서 제외 (`config.report_min_speed_kmh`), 차속 정보가 없으면 주행으로 간주합니다.
- 시각별 위험 시간대는 `data/report_history.csv`에 이벤트가 10개 이상 쌓이면 표시됩니다.
- 동작 확인(카메라 불필요): `python tests/test_session_report.py`
