"""Shared persistence logic for object relations; no hand detector dependency."""
class TemporalRelation:
    def __init__(self,hold_ms=1000,min_samples=3,max_gap_ms=2500,roi=(0,0,1,1)):
        self.hold_ms=hold_ms;self.min_samples=min_samples;self.max_gap_ms=max_gap_ms;self.roi=roi
        self.previous=None;self.last=None;self.since=0;self.samples=0
    def inside(self,p,w,h):
        x1,y1,x2,y2=self.roi
        return x1*w<=p[0]<=x2*w and y1*h<=p[1]<=y2*h
    def update(self,objects,face_result,width,height,timestamp_ms):
        faces=getattr(face_result,'face_landmarks',[]) or []
        raw,obj,reason=self.classify(objects,faces,width,height)
        gap=self.last is None or timestamp_ms-self.last>self.max_gap_ms or timestamp_ms<=self.last
        if gap or raw!=self.previous:self.since=timestamp_ms;self.samples=1
        else:self.samples+=1
        self.last=timestamp_ms;self.previous=raw;elapsed=timestamp_ms-self.since
        active=raw in ('holding','smoking')
        confirmed=active and elapsed>=self.hold_ms and self.samples>=self.min_samples
        return dict(state=raw if confirmed or not active else 'checking',candidate=raw,confirmed=confirmed,
                    reason=reason,elapsed_ms=elapsed,confidence=obj['confidence'] if obj else None)
