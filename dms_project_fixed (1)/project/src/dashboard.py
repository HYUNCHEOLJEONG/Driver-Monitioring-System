"""Live DMS dashboard. All metrics model with Ambient Lighting & Stable Layout."""
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .policy import korean_reason


class Dashboard:
    RED='#ff514c'; GREEN='#35dca4'; AMBER='#ffc34d'; MUTED='#8a9ca9'
    
    
    def __init__(self):
        self.result={'state':'no_phone'}
        self.smoking_result={'state':'unknown'}
        self.fonts={n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in (11,12,13,15,18,20,22,30)}
        
        self.drowsy_event_count = 0.0   
        self.last_event_time = 0        
        self.current_stage = 1          
        self.last_decay_time = 0        

    @staticmethod
    def analyze_ambient_lighting(frame):
        """프레임 밝기 및 적외선/조도 상태를 분석하여 낮/밤 및 햇빛 할당량 판별"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        mean_brightness = float(np.mean(gray))
        
        if mean_brightness < 45.0:
            return "NIGHT/IR", "낮음 (0%)", mean_brightness
        elif mean_brightness < 90.0:
            return "TUNNEL", "보통 (30~50%)", mean_brightness
        else:
            return "DAY", "충분 (100%)", mean_brightness

    @staticmethod
    def gaze_progress(gaze, valid):
        """차속별 허용 시간(limit) 기준 게이지. 저속(limit 없음)·예외 중에는 게이지를 띄우지 않는다."""
        info=gaze.info if gaze else {}
        limit=info.get('limit')
        active=bool(valid and info.get('face') and info.get('state') in ('head_left','head_right','head_up')
                    and limit and not info.get('paused'))
        duration=max(0.0,float(info.get('duration',0))) if active else 0.0
        return active, duration, min(1.0, duration/limit) if active else 0.0, active and gaze.alert=='warning'

    def early_card(self, early, valid):
        """초기 졸음 점수(early_drowsiness)를 3단계로만 표시: 7점 미만 정상, 7~10점 미만 주의, 10점 이상 경고."""
        head = 'EARLY DROWSINESS · 초기 졸음'
        if not (valid and early and early.value is not None):
            return (head, '확인 불가', self.MUTED)
        if early.value >= 10:
            return (head, '경고', self.RED)
        if early.value >= 7:
            return (head, '주의', self.AMBER)
        return (head, '정상', self.GREEN)

    def render(self, ctx, fps=0, camera_index=0, draw_landmarks=True):
        frame=ctx.frame.copy(); h, w=frame.shape[:2]
        
        # 조도 환경 분석
        env_state, sun_quota, brightness = self.analyze_ambient_lighting(frame)

        results=ctx.results
        eye=results.get('eye_state'); closure=results.get('eye_closure'); perclos=results.get('perclos'); gaze=results.get('gaze')
        fatigue = results.get('fatigue')
        
        t_ms = ctx.timestamp_ms
        
        vehicle_state = getattr(ctx, "vehicle", None) or results.get("vehicle")
        raw_speed = getattr(ctx, "speed_kmh", None)
        if raw_speed is None and vehicle_state:
            if hasattr(vehicle_state, "speed_kph") and vehicle_state.speed_kph is not None:
                raw_speed = vehicle_state.speed_kph
            elif isinstance(vehicle_state, dict):
                raw_speed = vehicle_state.get("speed_kph") if vehicle_state.get("speed_kph") is not None else vehicle_state.get("speed")

        speed = float(raw_speed) if raw_speed is not None else None

        faces=getattr(ctx.face_result,'face_landmarks',[]) or []
        valid=len(faces)==1
        closed=bool(valid and eye and eye.info.get('closed'))
        gaze_valid=valid and gaze and gaze.info.get('face')
        gaze_state=gaze.info.get('state','normal') if gaze_valid else 'unknown'
        if gaze_state=='alarm': gaze_state='head_down'
        head_down=gaze_valid and gaze_state=='head_down'
        calibrating=bool(gaze_valid and 'Calibrating' in gaze.info.get('message',''))
        offroad=gaze_valid and not calibrating and gaze_state!='normal'
        gaze_active, gaze_seconds, gaze_fill, gaze_warning=self.gaze_progress(gaze,valid)
        phone=self.result.get('state')=='holding'; smoking=self.smoking_result.get('state')=='smoking'
        
        rest_recommended = fatigue and fatigue.info.get('rest_recommended', False)
        
        closure_alert = closure.alert if closure else 'none'
        closure_val = closure.value if closure and closure.value is not None else 0.0
        perclos_val = perclos.value if perclos and perclos.value is not None else 0.0
        
        # [누적 감김 횟수 및 단계 상승]
        if closure_alert in ('warning', 'alarm'):
            if t_ms - self.last_event_time > 1500:
                self.drowsy_event_count += 1.0
                self.last_event_time = t_ms
                
                if self.drowsy_event_count >= 4.0:
                    self.current_stage = 3  # 3단계 (위험)
                elif self.drowsy_event_count >= 2.0:
                    self.current_stage = 2  # 2단계 (주의)
                else:
                    self.current_stage = 1  # 1단계 (안전)
                self.last_decay_time = t_ms 

        # [안전 주행 유지 시 자동 초기화 및 단계 하향 (Decay)]
        if self.current_stage > 1 and (t_ms - self.last_decay_time > 12000):
            if closure_alert == 'none' and perclos_val < 0.1:
                self.current_stage -= 1
                self.last_decay_time = t_ms
                if self.current_stage == 1:
                    self.drowsy_event_count = 0.0

        active_slots = self.current_stage
        
        stage_names = {1: '안전 주행 (1단계)', 2: '피로 주의 (2단계)', 3: '졸음 위험 (3단계)'}
        drowsy_stage_text = stage_names.get(active_slots, '안전 주행 (1단계)')

        drowsy=valid and any(r and r.alert in ('warning','alarm') for r in (closure,perclos))
        color=self.RED if drowsy or smoking or phone or head_down or gaze_warning or rest_recommended or active_slots == 3 else self.AMBER if offroad or not valid or active_slots == 2 else self.GREEN
        title=('HEAD DOWN ALERT' if head_down else 'DROWSINESS ALERT' if drowsy else 'SMOKING ALERT' if smoking else 'PHONE USE DETECTED' if phone
               else 'ATTENTION ALERT' if gaze_warning else 'CHECK FORWARD' if offroad else 'MONITORING ACTIVE' if valid else 'DRIVER NOT CONFIRMED')
        
        if draw_landmarks:
            for face in faces:
                for ids in ([33,160,158,133,153,144],[362,385,387,263,373,380],[61,13,291,14]):
                    pts=np.array([(int(face[i].x*w),int(face[i].y*h)) for i in ids])
                    x,y,bw,bh=cv2.boundingRect(pts)
                    cv2.rectangle(frame,(x-5,y-5),(x+bw+5,y+bh+5),(75,80,255) if closed or drowsy else (170,220,75),1)
                    for px,py in pts: cv2.circle(frame,(int(px),int(py)),2,(75,80,255),-1)
        for obj in ctx.objects:
            x1,y1,x2,y2=obj['bbox']; c=(255,210,0) if obj['label']=='cell_phone' else (180,90,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
            
        # 영상은 640폭으로 가운데 배치 + 아래 빈 공간 제거 -> 창(960x900)에 맞춰 줄어드는 비율이 작아져 하단 글자가 커 보임
        W=960; VW=640; vh=round(h*VW/w); H=vh+460
        
        canvas=Image.new('RGB',(W,H),'#060e15')
        video=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(VW,vh)),cv2.COLOR_BGR2RGB))
        canvas.paste(video,((W-VW)//2,52))
        d=ImageDraw.Draw(canvas)
        
        def text(x,y,s,size=15,fill='white'): d.text((x,y),str(s),font=self.fonts[size],fill=fill)
        
        # [생체 리듬 긴급 플래시 테두리 효과]
        if active_slots == 3:
            d.rectangle((0, 0, W-1, H-1), outline=(255, 81, 76), width=4)
        elif active_slots == 2:
            d.rectangle((0, 0, W-1, H-1), outline=(255, 195, 77), width=2)

        # [상단바] 속도 및 환경 조도 정보 표출
        env_label = f"ENV: {env_state} (햇빛: {sun_quota})"
        env_color = self.AMBER if env_state == "TUNNEL" else (self.MUTED if env_state == "NIGHT/IR" else self.GREEN)
        
        text(24,15,f'CAM {camera_index:02d}  /  DRIVER FACING  ·  {env_label}',13,env_color)
        text(690,15,f'{fps:.1f} FPS   {datetime.now():%H:%M:%S}',15,self.MUTED)
        
        badges=[]
        if valid: badges.append(('EYES CLOSED' if closed else 'EYES OPEN',self.RED if closed else self.GREEN))
        if offroad: badges.append(('HEAD: DOWN' if head_down else 'HEAD: OFF ROAD',self.RED if head_down or gaze_warning else self.AMBER))
        if not valid: badges.append(('FACE NOT CONFIRMED',self.AMBER))
        for i,(label,c) in enumerate(badges):
            y_badge=65+i*38; d.rounded_rectangle((22,y_badge,282,y_badge+30),6,fill='#07131d',outline=c); text(34,y_badge+4,label,15,c)
            
        # [우측 상단] 전방 미주 / 고개 숙임 감지 게이지
        if gaze_active or head_down:
            gauge_color=self.RED if gaze_warning or head_down else self.AMBER
            d.rounded_rectangle((510,65,938,167),10,fill='#07131d',outline=gauge_color,width=2)
            text(526,75,'고개 숙임 경고' if head_down else ('전방 미주시 경고' if gaze_warning else '전방 미주시 감지'),22,gauge_color)
            text(526,107,f'{gaze_seconds:.1f}초 / {gaze.info["limit"]:.1f}초' if gaze_active else '',15,'white')
            d.rounded_rectangle((526,140,922,151),4,fill='#243544')
            if gaze_fill>0:
                d.rounded_rectangle((526,140,526+396*gaze_fill,151),4,fill=gauge_color)
                
        # [순차적 상대 레이아웃 1단계] 고개 편차 바 위치 (영상 바로 밑)
        current_y = 52 + vh + 10
        if gaze_valid and not calibrating and isinstance(gaze.value,dict):
            pitch=gaze.value.get('pitch',0); yaw=gaze.value.get('yaw',0)
            threshold=gaze.info.get('pitch_threshold',18)
            d.rounded_rectangle((16, current_y, 730, current_y + 38), 6, fill='#07131d')
            text(26, current_y + 9, f'고개 편차  좌우 {yaw:+.1f}px / 상하 {pitch:+.1f}px · 숙임 기준 >{threshold:.1f}px · R 재보정', 15, self.MUTED)
            current_y += 46

        # [순차적 상대 레이아웃 2단계] 3단계 누적 피로도 바 위젯
        bar_section_y = current_y
        header_color = self.RED if active_slots == 3 else (self.AMBER if active_slots == 2 else self.GREEN)
        
        d.rounded_rectangle((16, bar_section_y, 944, bar_section_y + 58), 10, fill='#07131d', outline='#1c2d3a', width=1)
        text(28, bar_section_y + 10, f'누적 감김 횟수: {int(self.drowsy_event_count)}회  ·  {drowsy_stage_text}', 13, header_color)
        
        bar_w = 888; box_h = 7; start_x = 28; start_y_bar = bar_section_y + 34; gap = 12
        single_box_w = (bar_w - (gap * 2)) / 3
        
        for idx in range(3):
            bx = start_x + int(idx * (single_box_w + gap))
            slot_color = '#1a2935'
            if idx < active_slots:
                if idx == 0: slot_color = self.GREEN   
                elif idx == 1: slot_color = self.AMBER   
                else: slot_color = self.RED     
            d.rounded_rectangle((bx, start_y_bar, bx + int(single_box_w), start_y_bar + box_h), 3, fill=slot_color)

        current_y = bar_section_y + 68

        # [순차적 상대 레이아웃 3단계] 하단 메인 대시보드 패널
        dashboard_y = current_y
        d.rounded_rectangle((16, dashboard_y, 944, dashboard_y + 265), 14, fill='#060e15', outline=color, width=2)
        text(36, dashboard_y + 13, 'DRIVER MONITOR  ·  LIVE', 13, self.MUTED)
        text(36, dashboard_y + 34, title, 30, color)
        text(36, dashboard_y + 76, 'PERCLOS · 눈 감김 비율 / 30초', 15, self.MUTED)
        
        p = perclos.value if perclos and valid else None
        text(421, dashboard_y + 72, f'{p*100:.1f}%' if p is not None else '수집 중' if valid else '--', 22, color)
        for i in range(20):
            c_box = ('#35dca4' if i < 3 else '#ffc34d' if i < 6 else '#ff514c') if p is not None and i / 20 < p else '#1a2935'
            d.rounded_rectangle((36 + i * 23, dashboard_y + 105, 55 + i * 23, dashboard_y + 118), 3, fill=c_box)
            
        # 눈 감김 지속: 헤더 줄 오른쪽으로 올림
        dur = closure_val
        text(540, dashboard_y + 11, f"눈 감김 지속  {dur:.1f}초" if dur > 0 else '눈 감김 지속  --', 15,
             self.RED if closure_alert == 'alarm' else self.AMBER if closure_alert == 'warning' else self.MUTED)

        # [차량 입력(CAN/수동) 패널] - 하단 카드와 같은 테마. 윗줄 2칸, 아랫줄(예외)은 한 칸으로 길게
        v = vehicle_state
        on = lambda b: '●' if b else '○'
        turn_on = any(getattr(v, k, False) for k in ('turn_left', 'turn_right', 'hazard'))
        reason = korean_reason(ctx.exemptions.get('gaze', ''))
        text(540, dashboard_y + 40, 'VEHICLE', 12, self.MUTED)
        # 오른쪽 끝을 하단 4번째 카드(x=912)에 맞추고, 간격은 하단 카드와 같은 12px
        x0, x1, gap, tile_h = 540, 912, 12, 38
        half = (x1 - x0 - gap) // 2
        y0 = dashboard_y + 58; y1 = y0 + tile_h + 10
        tiles = [
            (x0, y0, half, '속도 · 기어', f'{"--" if speed is None else f"{speed:.0f}"} km/h  ·  {getattr(v, "gear", None) or "-"}', 'white'),
            (x0 + half + gap, y0, half, '방향지시등 · 비상등',
             f'좌 {on(getattr(v, "turn_left", False))}  우 {on(getattr(v, "turn_right", False))}  비상 {on(getattr(v, "hazard", False))}',
             self.AMBER if turn_on else 'white'),
            (x0, y1, x1 - x0, '시선 경고 예외',
             f'{reason}  ·  시선 이탈 경고 일시 중지' if reason else '없음  ·  시선 이탈 경고 정상 동작',
             self.AMBER if reason else self.MUTED),
        ]
        for tx, ty, tw, label, value, c in tiles:
            d.rounded_rectangle((tx, ty, tx + tw, ty + tile_h), 7, fill='#101f2b', outline='#203441')
            text(tx + 10, ty + 3, label, 11, self.MUTED)
            text(tx + 10, ty + 16, value, 15, c)

        # 하단 3개 카드 영역
        gaze_label = '확인 불가' if not gaze_valid else '보정 중' if calibrating else {'normal':'전방','head_left':'왼쪽','head_right':'오른쪽','head_up':'위쪽','head_down':'고개 떨굼'}.get(gaze_state, gaze_state)
        smoke_label = {'unknown':'확인 불가','no_cigarette':'미검출','cigarette_visible':'담배 검출','checking':'입 근처 · 확인 중','smoking':'흡연 추정'}.get(self.smoking_result.get('state'), '확인 불가')
        
        cards = [
            ('HEAD DIRECTION', gaze_label, self.RED if (head_down or gaze_warning) else self.AMBER if offroad else self.MUTED if not gaze_valid else self.GREEN),
            ('PHONE USE', '휴대폰 조작' if phone else '미검출', self.RED if phone else self.GREEN),
            ('SMOKING', smoke_label, self.RED if smoking else self.AMBER if smoke_label == '확인 불가' else self.GREEN),
            self.early_card(results.get('early_drowsiness'), valid),
        ]
               
        for i, (label, value, c) in enumerate(cards):
            cx = 36 + i * 222
            d.rounded_rectangle((cx, dashboard_y + 155, cx + 210, dashboard_y + 225), 9, fill='#101f2b', outline='#203441')
            text(cx + 14, dashboard_y + 164, label, 12, self.MUTED)
            text(cx + 14, dashboard_y + 186, value, 18, c)

        muted = gaze and gaze.info.get('bypassed')
        text(36, dashboard_y + 238, 'Q 종료  R 재보정  B 음소거' + (' ON' if muted else '') +
             '  |  0-3 속도  C 속도해제  G 후진  D 주행  A/L 방향지시등  H 비상등', 11, self.MUTED)
        
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)