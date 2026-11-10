"""Live DMS dashboard. All metrics model with Ambient Lighting & Stable Layout."""
from collections import deque
from datetime import datetime
import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Dashboard:
    RED='#ff514c'; GREEN='#35dca4'; AMBER='#ffc34d'; MUTED='#8a9ca9'
    
    _can_cache = None
    
    def __init__(self):
        self.result={'state':'no_phone'}
        self.smoking_result={'state':'unknown'}
        self.history=deque(maxlen=240); self.last_sample=-1000
        self.fonts={n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in (11,12,13,15,18,20,22,30)}
        
        self.drowsy_event_count = 0.0   
        self.last_event_time = 0        
        self.current_stage = 1          
        self.last_decay_time = 0        

    @classmethod
    def _load_can_speed(cls, timestamp_ms):
        """CAN ID 2C0 데이터 파싱 로직 정밀 보정 (실제 속도 변화 반영)"""
        if cls._can_cache is None:
            cls._can_cache = []
            try:
                csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_can.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 3 and row[1].strip() == '2C0':
                                try:
                                    t = float(row[0].strip())
                                    hex_data = row[2].strip().replace(" ", "")
                                    if len(hex_data) >= 4:
                                        # 표준 바이트 오더 적용 (앞 2바이트 조합 후 0.01 스케일)
                                        # A00F 같은 헥사 데이터를 바이트 스환하여 실제 속도 값 추출
                                        b1 = int(hex_data[0:2], 16)
                                        b2 = int(hex_data[2:4], 16)
                                        val = ((b2 << 8) | b1) * 0.01
                                        if 0.0 <= val <= 300.0:
                                            cls._can_cache.append((t, val))
                                except ValueError:
                                    continue
            except Exception:
                pass
        
        if not cls._can_cache:
            return 0.0
            
        current_sec = (timestamp_ms / 1000.0) % (cls._can_cache[-1][0] if cls._can_cache else 40.0)
        
        current_speed = 0.0
        for t, val in cls._can_cache:
            if t >= current_sec:
                current_speed = val
                break
        return current_speed if current_speed > 0 else cls._can_cache[0][1]

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
        info=gaze.info if gaze else {}
        active=bool(valid and info.get('face') and info.get('state') in ('head_left','head_right','head_up'))
        duration=max(0.0,float(info.get('duration',0))) if active else 0.0
        return active, duration, min(1.0, duration/3.5), active and duration>=3.5

    def render(self, ctx, fps=0, camera_index=0, draw_landmarks=True):
        frame=ctx.frame.copy(); h, w=frame.shape[:2]
        
        # 조도 환경 분석
        env_state, sun_quota, brightness = self.analyze_ambient_lighting(frame)

        results=ctx.results
        eye=results.get('eye_state'); closure=results.get('eye_closure'); perclos=results.get('perclos'); gaze=results.get('gaze')
        fatigue = results.get('fatigue')
        
        t_ms = ctx.timestamp_ms
        
        vehicle_state = getattr(ctx, "vehicle", None) or results.get("vehicle")
        speed = 0.0
        if vehicle_state:
            if hasattr(vehicle_state, "speed_kph") and vehicle_state.speed_kph is not None:
                speed = float(vehicle_state.speed_kph)
            elif isinstance(vehicle_state, dict):
                speed = float(vehicle_state.get("speed_kph") or vehicle_state.get("speed", 0.0))
        
        if speed <= 0.0:
            speed = self._load_can_speed(t_ms)

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
            
        W=960; vh=round(h*W/w)
        
        canvas=Image.new('RGB',(W,vh+500),'#060e15')
        video=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(W,vh)),cv2.COLOR_BGR2RGB))
        canvas.paste(video,(0,52))
        d=ImageDraw.Draw(canvas)
        
        def text(x,y,s,size=15,fill='white'): d.text((x,y),str(s),font=self.fonts[size],fill=fill)
        
        # [생체 리듬 긴급 플래시 테두리 효과]
        if active_slots == 3:
            d.rectangle((0, 0, W-1, vh+500-1), outline=(255, 81, 76), width=4)
        elif active_slots == 2:
            d.rectangle((0, 0, W-1, vh+500-1), outline=(255, 195, 77), width=2)

        # [상단바] 속도 및 환경 조도 정보 표출
        env_label = f"ENV: {env_state} (햇빛: {sun_quota})"
        env_color = self.AMBER if env_state == "TUNNEL" else (self.MUTED if env_state == "NIGHT/IR" else self.GREEN)
        
        text(24,15,f'CAM {camera_index:02d}  /  DRIVER FACING  ·  SPEED: {speed:.1f} km/h  ·  {env_label}',13,env_color)
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
            text(526,107,f'{gaze_seconds:.1f}초 / 3.5초',15,'white')
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
            
        if ctx.timestamp_ms - self.last_sample >= 250:
            self.history.append(p)
            self.last_sample = ctx.timestamp_ms
            
        text(540, dashboard_y + 76, 'PERCLOS TREND', 13, self.MUTED)
        d.line((540, dashboard_y + 122, 916, dashboard_y + 122), fill='#273947')
        prev = None
        for i, v in enumerate(self.history):
            point = (540 + i * 376 / 239, dashboard_y + 122 - v * 34) if v is not None else None
            if prev and point: d.line((prev, point), fill=self.RED, width=2)
            prev = point
            
        dur = closure_val
        text(540, dashboard_y + 44, f"눈 감김 지속  {dur:.1f}초" if dur > 0 else '눈 감김 지속  --', 15, self.MUTED)

        # 하단 3개 카드 영역
        gaze_label = '확인 불가' if not gaze_valid else '보정 중' if calibrating else {'normal':'전방','head_left':'왼쪽','head_right':'오른쪽','head_up':'위쪽','head_down':'고개 떨굼'}.get(gaze_state, gaze_state)
        smoke_label = {'unknown':'확인 불가','no_cigarette':'미검출','cigarette_visible':'담배 검출','checking':'입 근처 · 확인 중','smoking':'흡연 추정'}.get(self.smoking_result.get('state'), '확인 불가')
        
        cards = [
            ('HEAD DIRECTION', gaze_label, self.RED if (head_down or gaze_warning) else self.AMBER if offroad else self.MUTED if not gaze_valid else self.GREEN),
            ('PHONE USE', '휴대폰 조작' if phone else '미검출', self.RED if phone else self.GREEN),
            ('SMOKING', smoke_label, self.RED if smoking else self.AMBER if smoke_label == '확인 불가' else self.GREEN)
        ]
               
        for i, (label, value, c) in enumerate(cards):
            cx = 36 + i * 300
            d.rounded_rectangle((cx, dashboard_y + 155, cx + 288, dashboard_y + 225), 9, fill='#101f2b', outline='#203441')
            text(cx + 14, dashboard_y + 164, label, 12, self.MUTED)
            text(cx + 14, dashboard_y + 184, value, 20, c)

        muted = gaze and gaze.info.get('bypassed')
        text(36, dashboard_y + 238, 'Q / ESC 종료    R 전방 재보정    B 고개 경고 음소거' + (' · ON' if muted else ''), 11, self.MUTED)
        
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)