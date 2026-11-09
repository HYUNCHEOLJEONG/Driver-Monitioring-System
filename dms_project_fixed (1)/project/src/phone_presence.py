from .temporal_relation import TemporalRelation

class PhonePresence(TemporalRelation):
    """Requested rule: any detected phone in the frame counts as phone use."""
    def __init__(self):
        super().__init__(hold_ms=0, min_samples=1)

    def classify(self, objects, faces, w, h):
        phones=[o for o in objects if o.get('label')=='cell_phone']
        if not phones:return 'no_phone',None,'no_phone_detection'
        return 'holding',max(phones,key=lambda o:o['confidence']),'phone_detected'
