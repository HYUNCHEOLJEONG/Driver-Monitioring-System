"""경고음 단일 출력. 분석은 멈추지 않고 한 번에 한 경고음만 재생."""
import threading
try:
    import winsound
except ImportError:
    winsound = None

class WarningSound:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = -10**9
        self._thread = None
        self._closed = False

    def update(self, ctx):
        if self._closed or winsound is None: return
        choices = []
        eye = ctx.results.get("eye_closure")
        gaze = ctx.results.get("gaze")
        if eye and eye.alert != "none" and eye.info.get("beep", True):
            choices.append((3, 2000, eye.info.get("beep_interval_ms", 1000)))
        if (gaze and gaze.alert != "none" and not gaze.info.get("paused")
                and not gaze.info.get("bypassed")):
            choices.append((2 if gaze.alert == "alarm" else 1, 1200, 1000))
        if not choices: return
        _, freq, gap = max(choices)
        if ctx.timestamp_ms - self._last < gap: return
        if not self._lock.acquire(blocking=False): return
        self._last = ctx.timestamp_ms
        def play():
            try: winsound.Beep(freq, 300)
            except Exception: pass
            finally: self._lock.release()
        self._thread = threading.Thread(target=play, daemon=True)
        self._thread.start()

    def close(self):
        self._closed = True
        if self._thread is not None: self._thread.join(timeout=0.5)
