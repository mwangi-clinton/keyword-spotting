import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import sounddevice as sd
import webrtcvad
import torch
import librosa
import queue
# ─── Global state and queues ──────────────────────────────────────────────────
RUNNING = False
THREAD = None
RESULTS_QUEUE = queue.Queue()  # For passing results from inference thread to UI

class MicrophoneStreamer:
    def __init__(self, rate=16000, frame_ms=30):
        self.rate = rate
        self.frame_ms = frame_ms
        self.frame_len = int(rate * frame_ms/1000)  # e.g. 320 samples
        self.q = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        # indata.shape = (frames, channels)
        # convert to mono if needed, flatten
        mono = indata[:,0] if indata.ndim>1 else indata
        # put raw samples into our queue
        self.q.put(mono.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate = self.rate,
            channels = 1,
            blocksize = self.frame_len,
            callback = self._callback
        )
        self.stream.start()

    def read(self):
        """Blocking read of one frame (frame_ms long)"""
        return self.q.get()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()