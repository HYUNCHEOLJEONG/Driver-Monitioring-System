"""Verify actual frame delivery, not only whether Windows opened the device."""
import time
import cv2

class VerifiedCapture:
    def __init__(self,capture,first_frame):
        self.capture=capture;self.first_frame=first_frame
    def read(self):
        if self.first_frame is not None:
            frame=self.first_frame;self.first_frame=None;return True,frame
        return self.capture.read()
    def release(self):self.capture.release()

def open_camera(cfg):
    failures=[]
    for backend,name in ((cv2.CAP_DSHOW,'DSHOW'),(cv2.CAP_MSMF,'MSMF')):
        for configured in (True,False):
            mode=f'{cfg.frame_width}x{cfg.frame_height}' if configured else 'device default'
            print(f'Camera {cfg.camera_index}: trying {name}, {mode}',flush=True)
            cam=cv2.VideoCapture(cfg.camera_index,backend)
            accepted=False
            try:
                if not cam.isOpened():
                    failures.append(f'{name}/{mode}: open failed');continue
                if configured:
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH,cfg.frame_width)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg.frame_height)
                for _ in range(12):
                    ok,frame=cam.read()
                    if ok and frame is not None and frame.size:
                        accepted=True
                        h,w=frame.shape[:2]
                        print(f'Camera ready: index={cfg.camera_index}, backend={name}, frame={w}x{h}',flush=True)
                        return VerifiedCapture(cam,frame)
                    time.sleep(.08)
                failures.append(f'{name}/{mode}: no frames')
            finally:
                if not accepted:cam.release()
            time.sleep(.1)
    raise RuntimeError(f'Camera {cfg.camera_index} cannot deliver frames. Close Camera/AMCap/Teams and other DMS instances; '
                       'check the camera connection and camera_index in config.py. Tried: '+'; '.join(failures))
