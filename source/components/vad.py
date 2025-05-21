
import sounddevice as sd
import webrtcvad
import queue

# ─── Global state and queues ──────────────────────────────────────────────────
RUNNING = False
THREAD = None
RESULTS_QUEUE = queue.Queue()  # For passing results from inference thread to UI
class VAD:
    def __init__(self, sample_rate=16000, frame_ms=30, mode=3):
        """
        mode=0..3, higher = more aggressive filtering
        """
        self.vad = webrtcvad.Vad(mode)
        self.rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(sample_rate*frame_ms/1000)

    def is_speech(self, pcm_frame):
        """
        pcm_frame: a Python bytes or int16 numpy array of length frame_len
        return: True=voice present
        """
        # WebRTC wants 16-bit little endian bytes
        if isinstance(pcm_frame, bytes):
            data = pcm_frame
        else:
            # assume numpy array of float32 or int16
            arr16 = (pcm_frame * 32767).astype('int16') if pcm_frame.dtype!='int16' else pcm_frame
            data = arr16.tobytes()
        return self.vad.is_speech(data, sample_rate=self.rate)