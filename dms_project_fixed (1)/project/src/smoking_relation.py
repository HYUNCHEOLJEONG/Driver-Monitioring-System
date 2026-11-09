"""Cigarette + mouth proximity, sustained across fresh inference samples."""
import math
from .temporal_relation import TemporalRelation

class SmokingRelation(TemporalRelation):
    def __init__(self,hold_ms=1000,min_samples=3,max_gap_ms=2500,roi=(0,0,1,1),mouth_distance=0.20):
        super().__init__(hold_ms,min_samples,max_gap_ms,roi=roi)
        self.mouth_distance=mouth_distance

    def classify(self,objects,faces,w,h):
        cigarettes=[o for o in objects if o.get('label')=='cigarette' and self.inside(
            ((o['bbox'][0]+o['bbox'][2])/2,(o['bbox'][1]+o['bbox'][3])/2),w,h)]
        pts=[[(p.x*w,p.y*h) for p in face] for face in faces if len(face)>454 and self.inside((face[1].x*w,face[1].y*h),w,h)]
        if len(pts)!=1:return 'unknown',None,'face_missing_or_ambiguous'
        if not cigarettes:return 'no_cigarette',None,'no_cigarette_detection'
        face=pts[0];fw=max(p[0] for p in face)-min(p[0] for p in face)
        if fw<20:return 'unknown',None,'face_too_small'
        mouth=((face[13][0]+face[14][0])/2,(face[13][1]+face[14][1])/2)
        best=max(cigarettes,key=lambda o:o['confidence'])
        for cigarette in sorted(cigarettes,key=lambda o:o['confidence'],reverse=True):
            x1,y1,x2,y2=cigarette['bbox']
            def distance(point):
                x,y=point
                return math.hypot(max(x1-x,0,x-x2),max(y1-y,0,y-y2))
            near_mouth=distance(mouth)<=self.mouth_distance*fw
            if near_mouth:
                # Parent handles the same temporal confirmation machinery as phone holding.
                return 'smoking',cigarette,'cigarette_near_mouth'
        return 'cigarette_visible',best,'mouth_not_near'

