"""Terminal-only transition events. Missing evidence is not a normal-state event."""
import re
from datetime import datetime
from .policy import korean_reason

class EventLogger:
    def __init__(self,sink=None):
        self.states={}
        self.sink=sink or (lambda line:print(line,flush=True))

    def emit(self,code,message,level='INFO'):
        stamp=datetime.now().astimezone().isoformat(timespec='seconds')
        self.sink(f'[{stamp}] [{level}] [{code}] {message}')

    def transition(self,code,state,message,timestamp_ms,level='INFO',detail=''):
        previous=self.states.get(code)
        if previous and previous[0]==state:return
        suffix=f' | 이전={previous[0]}, 유지={(timestamp_ms-previous[1])/1000:.1f}초' if previous else ' | 초기 상태'
        self.states[code]=(state,timestamp_ms)
        self.emit(code,message+(f' | {detail}' if detail else '')+suffix,level)

    def update(self,ctx,phone,smoking):
        t=ctx.timestamp_ms;results=ctx.results
        count=len(getattr(ctx.face_result,'face_landmarks',[]) or [])
        face_state='detected' if count==1 else 'missing' if count==0 else 'multiple'
        self.transition('FACE',face_state,{'detected':'얼굴 확인','missing':'얼굴 미검출 · 얼굴 기반 판정 불가','multiple':'여러 얼굴 · 운전자 판정 불명확'}[face_state],t,
                        'INFO' if count==1 else 'WARN')
        ps=phone.get('state','unknown')
        self.transition('PHONE',ps,{'holding':'휴대폰 조작 감지','no_phone':'휴대폰 미검출'}.get(ps,'휴대폰 상태 확인 불가'),t,
                        'WARN' if ps=='holding' else 'INFO',self.confidence(phone))
        ss=smoking.get('state','unknown')
        self.transition('SMOKING',ss,{'unknown':'흡연 판정 불가','no_cigarette':'담배 미검출','cigarette_visible':'담배 검출 · 입 근처 조건 미충족',
            'checking':'입 근처 담배 · 지속 시간 확인 중','smoking':'흡연 추정 감지'}.get(ss,ss),t,'WARN' if ss=='smoking' else 'INFO',self.confidence(smoking))
        gaze=results.get('gaze');info=gaze.info if gaze else {}
        state=info.get('state','normal')
        if count!=1 or not info.get('face'):state='unknown'
        elif 'Calibrating' in info.get('message',''):state='calibrating'
        elif state=='alarm':state='head_down'
        elif state in ('head_left','head_right','head_up'):
            limit=info.get('limit')
            state+=':warning' if gaze.alert=='warning' else ':pending'  # 차속 기준·예외 반영된 분석기 판정
        labels={'unknown':'고개 방향 판정 불가','calibrating':'전방 기준 자세 보정 중','normal':'전방 자세',
                'head_down':'고개 떨굼 경고'}
        direction={'head_left':'왼쪽','head_right':'오른쪽','head_up':'위쪽'}
        if ':' in state:
            side,phase=state.split(':')
            message=f"전방 미주시 {f'경고 ({limit:.1f}초 이상)' if phase=='warning' else '감지 · 시간 누적 중'} · {direction[side]}"
        else:message=labels.get(state,state)
        self.transition('GAZE',state,message,t,'WARN' if state=='head_down' or state.endswith(':warning') else 'INFO')
        vehicle_key=(ctx.speed_kmh,ctx.gear,ctx.turn_signal)
        self.transition('VEHICLE',vehicle_key,f'모의 차량 입력 · 차속={ctx.speed_kmh} · 기어={ctx.gear} · 방향지시등={ctx.turn_signal}',t)
        self.transition('GAZE_MUTE',bool(info.get('bypassed')), '고개 방향 경고 음소거 ON' if info.get('bypassed') else '고개 방향 경고 음소거 OFF',t)
        reason=korean_reason(re.sub(r' [\d.]+s','',ctx.exemptions.get('gaze','')))  # 경과 초는 빼야 매 프레임 로그가 안 찍힘
        self.transition('EXEMPT',reason,f'시선 경고 예외 적용 · {reason}' if reason else '시선 경고 예외 없음',t)
        yawn=results.get('yawn')
        if yawn and count==1:
            on=yawn.info.get('state')=='yawning'
            self.transition('YAWN',on,'하품 감지' if on else '하품 없음',t,'WARN' if on else 'INFO',
                            f"10분 내 {yawn.info.get('count',0)}회" if on else '')
        eye=results.get('eye_closure');perclos=results.get('perclos')
        for code,r,label in (('EYE_CLOSURE',eye,'눈 감김'),('PERCLOS',perclos,'눈 감김 비율')):
            state='unknown' if count!=1 or r is None else 'collecting' if r.value is None else r.alert
            message=f'{label} '+{'unknown':'판정 불가','collecting':'데이터 수집 중','none':'경고 없음','warning':'주의','alarm':'경고'}[state]
            detail=''
            if state not in ('unknown','collecting'):
                detail=f'{r.value:.1f}초' if code=='EYE_CLOSURE' else f'{r.value*100:.1f}%'
            self.transition(code,state,message,t,'WARN' if state in ('warning','alarm') else 'INFO',detail)

    @staticmethod
    def confidence(result):
        value=result.get('confidence')
        return f'객체 검출 신뢰도={value:.1%}' if value is not None else ''
