"""주행 세션 리포트: 하품 / PERCLOS / 휴대폰 / 흡연 수치를 모아 메일 제목 + HTML 보고서를 만든다.

- 주행 중(차속 >= min_speed_kmh, 차속 정보가 없으면 주행으로 간주)일 때만 집계한다.
- 이벤트는 '상태가 새로 시작될 때' 1회로 센다 (30fps 에서 하품 1번이 30번으로 세지지 않게).
- 조심할 시간대: 이번 주행은 15분 구간별, 시각(시)별은 history_path 에 누적된 과거 주행까지 합쳐 계산한다.
"""
import csv
import html
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

KINDS = {"yawn": "하품", "perclos": "PERCLOS 경고", "phone": "휴대폰", "smoking": "흡연"}
BIN_MIN = 15
# 일반적으로 졸음 사고가 잦은 시간대 (이력이 쌓이기 전 안내용)
RISK_HOURS = ((2, 6, "새벽 2~6시"), (14, 16, "오후 2~4시"))
MIN_HISTORY_EVENTS = 10


class SessionReport:
    def __init__(self, min_speed_kmh=10.0, history_path=None, start=None):
        self.min_speed = min_speed_kmh
        self.history_path = Path(history_path) if history_path else None
        self.start_wall = start or datetime.now()
        self._t0 = self._last_t = None
        self._prev = {}                       # kind -> 직전 프레임 활성 여부
        self.drive_sec = 0.0
        self.durations = Counter()            # kind -> 활성 시간(초)
        self.events = []                      # (발생 시각 datetime, kind, 그때 차속)
        self.perclos_sum = self.perclos_sec = 0.0
        self.perclos_max = None

    def _wall(self, t_ms):
        return self.start_wall + timedelta(milliseconds=t_ms - self._t0)

    def update(self, ctx, phone, smoking):
        t = ctx.timestamp_ms
        if self._t0 is None:
            self._t0 = t
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0) / 1000
        self._last_t = t
        speed = ctx.speed_kmh
        if speed is not None and speed < self.min_speed:
            self._prev.clear()                # 정차 중엔 집계하지 않음, 다시 출발하면 새 이벤트로
            return
        self.drive_sec += dt

        r = ctx.results
        yawn, perclos = r.get("yawn"), r.get("perclos")
        p = perclos.value if perclos else None
        if p is not None:
            self.perclos_sum += p * dt
            self.perclos_sec += dt
            self.perclos_max = p if self.perclos_max is None else max(self.perclos_max, p)

        active = {
            "yawn": bool(yawn and yawn.info.get("state") == "yawning"),
            "perclos": bool(perclos and perclos.alert == "alarm"),
            "phone": (phone or {}).get("state") == "holding",
            "smoking": (smoking or {}).get("state") == "smoking",
        }
        for kind, on in active.items():
            if on:
                self.durations[kind] += dt
                if not self._prev.get(kind):
                    self.events.append((self._wall(t), kind, speed))
            self._prev[kind] = on

    # ── 집계 ────────────────────────────────────────────────
    def counts(self):
        return Counter(k for _, k, _ in self.events)

    def stats(self):
        c, hours = self.counts(), self.drive_sec / 3600
        return {
            "drive_min": self.drive_sec / 60,
            "yawn": c["yawn"], "yawn_per_h": c["yawn"] / hours if hours >= 0.25 else None,
            "perclos_avg": self.perclos_sum / self.perclos_sec if self.perclos_sec else None,
            "perclos_max": self.perclos_max,
            "perclos_alarm": c["perclos"],
            "phone": c["phone"], "phone_sec": self.durations["phone"],
            "smoking": c["smoking"], "smoking_sec": self.durations["smoking"],
            "fast_phone": sum(1 for _, k, s in self.events if k == "phone" and s is not None and s >= 80),
        }

    def notes(self, s):
        """기준을 넘은 항목만, 심각한 순서대로."""
        out = []
        if s["fast_phone"]:
            out.append(f"80km/h 이상 고속 주행 중 휴대폰 사용이 {s['fast_phone']}회 감지되었습니다. 가장 위험한 행동입니다.")
        if s["perclos_max"] is not None and s["perclos_max"] >= 0.30:
            out.append(f"눈 감김 비율이 위험 수준(최대 {s['perclos_max']:.0%})까지 올라갔습니다. 운전 전 충분한 수면이 필요합니다.")
        elif s["perclos_avg"] is not None and s["perclos_avg"] >= 0.15:
            out.append(f"평균 눈 감김 비율이 {s['perclos_avg']:.0%}로 높습니다. 피로가 누적된 상태입니다.")
        if s["phone"] and not s["fast_phone"]:
            out.append(f"주행 중 휴대폰 사용 {s['phone']}회(총 {s['phone_sec']:.0f}초). 거치대·음성 기능을 이용하세요.")
        if s["yawn_per_h"] is not None and s["yawn_per_h"] >= 3:
            out.append(f"하품이 시간당 {s['yawn_per_h']:.1f}회로 잦습니다. 졸음의 초기 신호이니 휴식을 계획하세요.")
        elif s["yawn"] >= 3:
            out.append(f"하품이 {s['yawn']}회 감지되었습니다. 졸음의 초기 신호입니다.")
        if s["smoking"]:
            out.append(f"주행 중 흡연 {s['smoking']}회(총 {s['smoking_sec']:.0f}초). 한 손 운전과 시야 방해에 유의하세요.")
        return out or ["특별히 위험한 패턴은 감지되지 않았습니다. 안전 운전 습관을 유지하세요."]

    def bins(self):
        """이번 주행의 15분 구간별 이벤트 수."""
        n = int(self.drive_sec // (BIN_MIN * 60)) + 1
        b = [0] * n
        for w, _, _ in self.events:
            i = int((w - self.start_wall).total_seconds() // (BIN_MIN * 60))
            b[min(i, n - 1)] += 1
        return b

    def time_advice(self, history_hours=None):
        out = []
        b = self.bins()
        if sum(b) >= 2 and len(b) >= 2:
            i = max(range(len(b)), key=b.__getitem__)
            out.append(f"이번 주행은 시작 {i * BIN_MIN}~{(i + 1) * BIN_MIN}분 구간에 위험 신호가 가장 많았습니다."
                       + (f" {max(i * BIN_MIN, 30)}분마다 휴식을 권장합니다." if i else ""))
        hours = Counter(history_hours or [])
        hours.update(w.hour for w, _, _ in self.events)
        if sum(hours.values()) >= MIN_HISTORY_EVENTS:
            h, n = hours.most_common(1)[0]
            out.append(f"누적 기록상 {h}시~{h + 1}시에 위험 신호가 가장 많았습니다({n}회). 이 시간대 운전에 특히 유의하세요.")
        # 시작~끝 사이에 지나간 모든 시각(시)을 확인 (1시 출발 7시 도착도 2~6시에 걸림)
        end = self.start_wall + timedelta(seconds=self.drive_sec)
        driven = {(self.start_wall + timedelta(hours=i)).hour
                  for i in range(int(self.drive_sec // 3600) + 1)} | {end.hour}
        for lo, hi, label in RISK_HOURS:
            if any(lo <= h < hi for h in driven):
                out.append(f"{label}는 생체리듬상 졸음이 몰리는 시간대입니다.")
        return out

    # ── 이력 파일 ───────────────────────────────────────────
    def load_history_hours(self):
        if not self.history_path or not self.history_path.is_file():
            return []
        with open(self.history_path, encoding="utf-8") as f:
            return [datetime.fromisoformat(row["time"]).hour for row in csv.DictReader(f)]

    def append_history(self):
        if not self.history_path or not self.events:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.history_path.is_file()
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "kind", "speed_kmh"])
            w.writerows((t.isoformat(timespec="seconds"), k, "" if s is None else s) for t, k, s in self.events)

    # ── 출력 ────────────────────────────────────────────────
    def mail_subject(self, s):
        perclos = f"{s['perclos_max']:.0%}" if s["perclos_max"] is not None else "-"
        return (f"[DMS] {self.start_wall:%m/%d %H:%M} {s['drive_min']:.0f}분 주행 · "
                f"하품{s['yawn']} PERCLOS최대{perclos} 폰{s['phone']} 흡연{s['smoking']}")

    def html(self, s, notes, advice):
        e = html.escape
        pct = lambda v: "-" if v is None else f"{v:.1%}"
        rows = [
            ("주행 시간", f"{s['drive_min']:.0f}분"),
            ("하품", f"{s['yawn']}회" + (f" (시간당 {s['yawn_per_h']:.1f}회)" if s["yawn_per_h"] is not None else "")),
            ("PERCLOS 평균 / 최대", f"{pct(s['perclos_avg'])} / {pct(s['perclos_max'])}"),
            ("PERCLOS 위험(30%↑) 진입", f"{s['perclos_alarm']}회"),
            ("휴대폰 사용", f"{s['phone']}회 · {s['phone_sec']:.0f}초"),
            ("흡연", f"{s['smoking']}회 · {s['smoking_sec']:.0f}초"),
        ]
        b = self.bins()
        top = max(b) or 1
        bars = "".join(
            f'<div class="bar"><span>{i * BIN_MIN}~{(i + 1) * BIN_MIN}분</span>'
            f'<i style="width:{n / top * 100:.0f}%"></i><b>{n}</b></div>' for i, n in enumerate(b))
        ev = "".join(f"<li>{t:%H:%M:%S} · {KINDS[k]}" + (f" · {sp:.0f}km/h" if sp is not None else "") + "</li>"
                     for t, k, sp in self.events) or "<li>없음</li>"
        li = lambda xs: "".join(f"<li>{e(x)}</li>" for x in xs)
        return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>DMS 주행 리포트</title><style>
body{{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#fff;color:#222}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:24px}}table{{width:100%;border-collapse:collapse}}
td{{padding:8px;border-bottom:1px solid #ddd}}td:last-child{{text-align:right;font-weight:600}}
.bar{{display:flex;align-items:center;gap:8px;margin:4px 0}}.bar span{{width:80px;font-size:13px}}
.bar i{{display:block;height:14px;background:#e4572e;border-radius:3px;min-width:2px}}
ul{{padding-left:20px}}li{{margin:4px 0}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}td{{border-color:#333}}}}
</style></head><body>
<h1>DMS 주행 리포트</h1><p>{self.start_wall:%Y-%m-%d %H:%M} 시작</p>
<h2>수치</h2><table>{''.join(f'<tr><td>{a}</td><td>{e(v)}</td></tr>' for a, v in rows)}</table>
<h2>유의할 점</h2><ul>{li(notes)}</ul>
<h2>조심할 시간대</h2><ul>{li(advice) or '<li>판단할 데이터가 부족합니다.</li>'}</ul>
<h2>15분 구간별 위험 신호</h2>{bars}
<h2>이벤트 기록</h2><ul>{ev}</ul>
</body></html>"""

    def build(self):
        """-> (stats, notes, advice, html)"""
        s = self.stats()
        notes = self.notes(s)
        advice = self.time_advice(self.load_history_hours())
        return s, notes, advice, self.html(s, notes, advice)
