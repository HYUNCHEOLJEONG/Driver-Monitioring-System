"""주행 종료 시 보고서 HTML 저장 + 이메일 전송 (본문 = HTML 보고서, 같은 파일 첨부).

환경변수 (코드에 직접 쓰지 말 것)
  DMS_SMTP_USER      보내는 메일 주소 (예: xxx@gmail.com)
  DMS_SMTP_PASSWORD  앱 비밀번호 (Gmail: 2단계 인증 후 '앱 비밀번호' 16자리, 실제 비밀번호 아님)
  DMS_MAIL_TO        받는 주소 (쉼표로 여러 명 가능)
  DMS_SMTP_HOST      선택, 기본 smtp.gmail.com (네이버: smtp.naver.com)
  DMS_SMTP_PORT      선택, 기본 465 (SSL)
필수 변수가 하나라도 없으면 메일은 건너뛰고 파일만 저장한다.
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

MAIL_VARS = ("DMS_SMTP_USER", "DMS_SMTP_PASSWORD", "DMS_MAIL_TO")


def save_html(html, reports_dir, start):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"DMS_report_{start:%Y%m%d_%H%M%S}.html"
    path.write_text(html, encoding="utf-8")
    return path


def send_email(subject, text, html, filename):
    user = os.environ["DMS_SMTP_USER"]
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, os.environ["DMS_MAIL_TO"]
    msg.set_content(text)                         # HTML 을 못 보는 메일 앱용
    msg.add_alternative(html, subtype="html")
    msg.add_attachment(html.encode("utf-8"), maintype="text", subtype="html", filename=filename)
    host = os.environ.get("DMS_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("DMS_SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=15) as smtp:
        smtp.login(user, os.environ["DMS_SMTP_PASSWORD"])
        smtp.send_message(msg)


def finish_session(report, reports_dir, log=print):
    """보고서 저장 -> 이력 누적 -> 메일 전송. 어느 단계가 실패해도 예외를 밖으로 던지지 않는다."""
    if report.drive_sec <= 0:
        log("[REPORT] 주행 기록이 없어 보고서를 만들지 않습니다.")
        return None
    try:
        s, notes, advice, html = report.build()
        path = save_html(html, reports_dir, report.start_wall)
    except Exception as e:  # 종료 처리(카메라/모델 해제)를 막지 않도록
        log(f"[REPORT] 보고서 생성 실패: {e}")
        return None
    log(f"[REPORT] 보고서 저장: {path}")
    try:
        report.append_history()
    except OSError as e:
        log(f"[REPORT] 이력 저장 실패: {e}")

    subject = report.mail_subject(s)
    text = "\n".join([subject, "", *notes, "", *advice])
    missing = [k for k in MAIL_VARS if not os.environ.get(k)]
    if missing:
        log(f"[REPORT] 메일 전송 건너뜀 (환경변수 없음: {', '.join(missing)})\n{text}")
        return path
    try:
        send_email(subject, text, html, path.name)
        log(f"[REPORT] 메일 전송 완료: {os.environ['DMS_MAIL_TO']}")
    except Exception as e:  # 네트워크/인증 등 어떤 실패도 종료를 막지 않게
        log(f"[REPORT] 메일 전송 실패: {e}")
    return path
