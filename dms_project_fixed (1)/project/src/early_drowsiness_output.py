"""초기 졸음 전용 로그·추가 상태 패널. 기존 EventLogger/화면 배치는 유지한다."""
import numpy as np
from PIL import Image, ImageDraw


class EarlyDrowsinessOutput:
    def update(self, ctx, events):
        result = ctx.results.get("early_drowsiness")
        if result is None:
            return
        info = result.info
        for key, code, label in (("long_event", "LONG_EYE_CLOSURE", "긴 눈감김 완료"),
                                 ("yawn_event", "YAWN", "하품 감지"),
                                 ("nod_event", "HEAD_NOD", "고개 끄덕임 감지")):
            if info.get(key):
                events.emit(code, label + " | " + info["reason"])
        labels = {"normal": "초기 졸음 기준 미충족", "pending": "초기 졸음 후보 · 3초 확인 중",
                  "warning": "초기 졸음 의심 · 휴식 필요", "collecting": "초기 졸음 데이터 수집 중",
                  "unavailable": "초기 졸음 판정 보류 · 얼굴/눈 데이터 확인"}
        state = info["state"]
        events.transition("EARLY_DROWSINESS", state,
                          f"{labels[state]} | 점수={result.value:.1f}/{info['threshold']:g}",
                          ctx.timestamp_ms, "WARN" if result.alert != "none" else "INFO",
                          info["reason"])


def render_early_drowsiness(frame, ctx, fonts):
    result = ctx.results.get("early_drowsiness")
    if result is None:
        return frame
    i = result.info
    labels = {"normal": "관찰 중", "pending": "초기 졸음 후보", "warning": "초기 졸음 의심 · 휴식 필요",
              "collecting": "데이터 수집 중", "unavailable": "판정 보류"}
    color = '#ffc34d' if result.alert != 'none' else '#8a9ca9' if not i['ready'] else '#35dca4'
    strip = Image.new('RGB', (frame.shape[1], 96), '#07131d')
    d = ImageDraw.Draw(strip)
    d.line((16, 0, frame.shape[1]-16, 0), fill=color, width=2)
    d.text((24, 8), f"초기 졸음  {result.value:.1f} / {i['threshold']:g}점  ·  {labels[i['state']]}", font=fonts[20], fill=color)
    p = f"{i['perclos']:.1%}" if i['perclos'] is not None else '--'
    parts = i['parts']
    formula = f"({parts['perclos']:g} + {parts['long_closure']:g}) × {i['eye_multiplier']:g} + {parts['yawn']:g} + {parts['nod']:g}"
    d.text((24, 39), f"PERCLOS {p}  ·  긴 눈감김 {i['long_count']}회/분  ·  하품 {i['yawn_count']}회/3분  ·  끄덕임 {i['nod_count']}회/분", font=fonts[13], fill='white')
    d.text((24, 64), f"점수 {formula}  ·  유효 관측 {i['valid_sec']:.1f}/30초  ·  시험용 기준", font=fonts[13], fill='#8a9ca9')
    # 원본 대시보드 픽셀은 그대로 두고 하단에만 덧붙인다.
    return np.concatenate((frame, np.asarray(strip)[:, :, ::-1]), axis=0)
