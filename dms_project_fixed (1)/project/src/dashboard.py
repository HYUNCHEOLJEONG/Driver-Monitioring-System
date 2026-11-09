"""Live DMS dashboard. All metrics come from current analyzer results."""
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

class Dashboard:
    RED='#ff514c';GREEN='#35dca4';AMBER='#ffc34d';MUTED='#8a9ca9'
    def __init__(self):
        self.result={'state':'no_phone'}
        self.smoking_result={'state':'unknown'}
        self.history=deque(maxlen=240);self.last_sample=-1000
        self.fonts={n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in (13,15,18,22,30)}

    @staticmethod
    def gaze_progress(gaze, valid):
        info=gaze.info if gaze else {}
        active=bool(valid and info.get('face') and info.get('state') in ('head_left','head_right','head_up'))
        duration=max(0.0,float(info.get('duration',0))) if active else 0.0
        return active,duration,min(1.0,duration/3.5),active and duration>=3.5

    def render(self,ctx,fps=0,camera_index=0,draw_landmarks=True):
        frame=ctx.frame.copy();h,w=frame.shape[:2]
        results=ctx.results
        eye=results.get('eye_state');closure=results.get('eye_closure');perclos=results.get('perclos');gaze=results.get('gaze')
        faces=getattr(ctx.face_result,'face_landmarks',[]) or []
        valid=len(faces)==1
        closed=bool(valid and eye and eye.info.get('closed'))
        gaze_valid=valid and gaze and gaze.info.get('face')
        gaze_state=gaze.info.get('state','normal') if gaze_valid else 'unknown'
        if gaze_state=='alarm':gaze_state='head_down'  # Legacy gaze analyzer's downward-head state
        head_down=gaze_valid and gaze_state=='head_down'
        calibrating=bool(gaze_valid and 'Calibrating' in gaze.info.get('message',''))
        offroad=gaze_valid and not calibrating and gaze_state!='normal'
        gaze_active,gaze_seconds,gaze_fill,gaze_warning=self.gaze_progress(gaze,valid)
        phone=self.result.get('state')=='holding';smoking=self.smoking_result.get('state')=='smoking'
        drowsy=valid and any(r and r.alert in ('warning','alarm') for r in (closure,perclos))
        color=self.RED if drowsy or smoking or phone or head_down or gaze_warning else self.AMBER if offroad or not valid else self.GREEN
        title=('HEAD DOWN ALERT' if head_down else 'DROWSINESS ALERT' if drowsy else 'SMOKING ALERT' if smoking else 'PHONE USE DETECTED' if phone
               else 'ATTENTION ALERT' if gaze_warning else 'CHECK FORWARD' if offroad else 'MONITORING ACTIVE' if valid else 'DRIVER NOT CONFIRMED')
        if draw_landmarks:
            for face in faces:
                for ids in ([33,160,158,133,153,144],[362,385,387,263,373,380],[61,13,291,14]):
                    pts=np.array([(int(face[i].x*w),int(face[i].y*h)) for i in ids])
                    x,y,bw,bh=cv2.boundingRect(pts)
                    cv2.rectangle(frame,(x-5,y-5),(x+bw+5,y+bh+5),(75,80,255) if closed or drowsy else (170,220,75),1)
                    for px,py in pts:cv2.circle(frame,(int(px),int(py)),2,(75,80,255),-1)
        for obj in ctx.objects:
            x1,y1,x2,y2=obj['bbox'];c=(255,210,0) if obj['label']=='cell_phone' else (180,90,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
        W=960;vh=round(h*W/w)
        video=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(W,vh)),cv2.COLOR_BGR2RGB))
        canvas=Image.new('RGB',(W,vh+352),'#060e15');canvas.paste(video,(0,52));d=ImageDraw.Draw(canvas)
        def text(x,y,s,size=15,fill='white'):d.text((x,y),str(s),font=self.fonts[size],fill=fill)
        text(24,15,f'CAM {camera_index:02d}  /  DRIVER FACING',15,self.MUTED)
        text(690,15,f'{fps:.1f} FPS   {datetime.now():%H:%M:%S}',15,self.MUTED)
        badges=[]
        if valid:badges.append(('EYES CLOSED' if closed else 'EYES OPEN',self.RED if closed else self.GREEN))
        if offroad:badges.append(('HEAD: DOWN' if head_down else 'HEAD: OFF ROAD',self.RED if head_down or gaze_warning else self.AMBER))
        if not valid:badges.append(('FACE NOT CONFIRMED',self.AMBER))
        for i,(label,c) in enumerate(badges):
            y=70+i*38;d.rounded_rectangle((22,y,282,y+30),6,fill='#07131d',outline=c);text(34,y+4,label,15,c)
        if gaze_active:
            gauge_color=self.RED if gaze_warning else self.AMBER
            d.rounded_rectangle((510,70,938,172),10,fill='#07131d',outline=gauge_color,width=2)
            text(526,80,'전방 미주시 경고' if gaze_warning else '전방 미주시 감지',22,gauge_color)
            text(526,112,f'{gaze_seconds:.1f}초 / 3.5초',15,'white')
            d.rounded_rectangle((526,145,922,156),4,fill='#243544')
            if gaze_fill>0:
                d.rounded_rectangle((526,145,526+396*gaze_fill,156),4,fill=gauge_color)
        if gaze_valid and not calibrating and isinstance(gaze.value,dict):
            pitch=gaze.value.get('pitch',0);yaw=gaze.value.get('yaw',0)
            threshold=gaze.info.get('pitch_threshold',18)
            d.rounded_rectangle((22,52+vh-40,730,52+vh-8),6,fill='#07131d')
            text(34,52+vh-35,f'고개 편차  좌우 {yaw:+.1f}px / 상하 {pitch:+.1f}px · 숙임 기준 >{threshold:.1f}px · R 재보정',15,self.MUTED)
        y=vh+66
        d.rounded_rectangle((16,y,944,y+266),14,fill='#07121c',outline=color,width=2)
        text(36,y+13,'DRIVER MONITOR  ·  LIVE',13,self.MUTED);text(36,y+34,title,30,color)
        text(36,y+86,'PERCLOS · 눈 감김 비율 / 30초',15,self.MUTED)
        p=perclos.value if perclos and valid else None
        text(421,y+82,f'{p*100:.1f}%' if p is not None else '수집 중' if valid else '--',22,color)
        for i in range(20):
            c=('#35dca4' if i<3 else '#ffc34d' if i<6 else '#ff514c') if p is not None and i/20<p else '#1a2935'
            d.rounded_rectangle((36+i*23,y+119,55+i*23,y+134),3,fill=c)
        if ctx.timestamp_ms-self.last_sample>=250:
            self.history.append(p);self.last_sample=ctx.timestamp_ms
        text(540,y+85,'PERCLOS TREND',13,self.MUTED)
        d.line((540,y+140,916,y+140),fill='#273947')
        prev=None
        for i,v in enumerate(self.history):
            point=(540+i*376/239,y+140-v*34) if v is not None else None
            if prev and point:d.line((prev,point),fill=self.RED,width=2)
            prev=point
        dur=closure.value if closure and valid else None
        text(540,y+53,f"눈 감김 지속  {dur:.1f}초" if dur is not None else '눈 감김 지속  --',15,self.MUTED)
        gaze_label='확인 불가' if not gaze_valid else '보정 중' if calibrating else {'normal':'전방','head_left':'왼쪽','head_right':'오른쪽','head_up':'위쪽','head_down':'고개 떨굼'}.get(gaze_state,gaze_state)
        smoke_label={'unknown':'확인 불가','no_cigarette':'미검출','cigarette_visible':'담배 검출','checking':'입 근처 · 확인 중','smoking':'흡연 추정'}.get(self.smoking_result.get('state'),'확인 불가')
        cards=[('HEAD DIRECTION',gaze_label,self.RED if head_down or gaze_warning else self.AMBER if offroad else self.MUTED if not gaze_valid else self.GREEN),
               ('PHONE USE','휴대폰 조작' if phone else '미검출',self.RED if phone else self.GREEN),
               ('SMOKING',smoke_label,self.RED if smoking else self.AMBER if smoke_label=='확인 불가' else self.GREEN)]
        for i,(label,value,c) in enumerate(cards):
            x=36+i*298;d.rounded_rectangle((x,y+160,x+280,y+229),9,fill='#101f2b',outline='#203441')
            text(x+14,y+171,label,13,self.MUTED);text(x+14,y+191,value,22,c)
        muted=gaze and gaze.info.get('bypassed')
        text(36,y+241,'Q / ESC 종료    R 전방 재보정    B 고개 경고 음소거'+(' · ON' if muted else ''),13,self.MUTED)
        return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
