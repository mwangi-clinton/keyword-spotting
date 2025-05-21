import queue
import os
import time
import threading
import numpy as np

from source.components.data_ingestion import MicrophoneStreamer
from source.components.vad import VAD
from source.components.predict import LiveKeywordSpotter
# ─── Global state and queues ──────────────────────────────────────────────────
RUNNING = False
THREAD = None
RESULTS_QUEUE = queue.Queue()  # For passing results from inference thread to UI
class InferenceThread(threading.Thread):
    def __init__(self, model_path='/home/clinton-mwangi/Desktop/msc-telecom/1.2/Speech/kws/keyword-spotting/source/components/model1.pth'):
        super().__init__()
        self.mic = MicrophoneStreamer(rate=16000, frame_ms=30)
        self.vad = VAD(sample_rate=16000, frame_ms=30, mode=3)  
        self.spotter = LiveKeywordSpotter(model_path=model_path)
        self.running = False
        self.daemon = True  
        
    def run(self):
        self.running = True
        self.mic.start()
        buffer = []
        speech_started = False
        silence_frames = 0
        
        # Calculate frames needed for 0.5 second (reduced from 1 second for faster inference)
        frames_per_half_second = int(500 / 30)  # 30ms frames = ~17 frames per half second
        
        try:
            print("Inference thread started, listening...")
            while self.running:
                frame = self.mic.read()  # 30 ms samples, numpy float32

                
                is_speech = self.vad.is_speech(frame)
                
                if is_speech:
                    if not speech_started:
                        print("Speech detected!")
                        RESULTS_QUEUE.put({
                            'timestamp': time.strftime('%H:%M:%S'),
                            'label': "Speech detected...",
                            'index': -1
                        })
                    
                    buffer.append(frame)
                    speech_started = True
                    silence_frames = 0
                    
                    # Process after collecting ~0.5 second of speech (faster response)
                    if len(buffer) >= frames_per_half_second:
                        print(f"Processing speech: {len(buffer)} frames")
                        audio = np.concatenate(buffer)
                        pred_idx, pred_label, confidence = self.spotter.predict(audio)
                        
                        # Format confidence as percentage with 2 decimal places
                        confidence_pct = confidence * 100
                        
                        # Only add to results if confidence is above threshold (0.3 = 30%)
                        if confidence > 0.3:
                            result_label = f"{pred_label} ({confidence_pct:.2f}%)"
                            print(f"Predicted: {result_label}")
                            
                            RESULTS_QUEUE.put({
                                'timestamp': time.strftime('%H:%M:%S'),
                                'label': result_label,
                                'index': pred_idx,
                                'confidence': confidence
                            })
                        
                        # Keep the last 5 frames (150ms) to maintain context but allow faster updates
                        buffer = buffer[-5:] if len(buffer) > 5 else buffer
                else:
                    if speech_started:
                        silence_frames += 1
                    
                    # End of utterance after 5 frames (150ms) of silence (reduced for faster response)
                    if speech_started and silence_frames > 5:
                        if len(buffer) > 10:  # At least 300ms of speech
                            print(f"Processing end of utterance: {len(buffer)} frames")
                            audio = np.concatenate(buffer)
                            pred_idx, pred_label, confidence = self.spotter.predict(audio)
                            
                            # Format confidence as percentage with 2 decimal places
                            confidence_pct = confidence * 100
                            
                            # Only add to results if confidence is above threshold
                            if confidence > 0.3:  # 30% confidence threshold
                                result_label = f"{pred_label} ({confidence_pct:.2f}%)"
                                print(f"Predicted: {result_label}")
                                
                                RESULTS_QUEUE.put({
                                    'timestamp': time.strftime('%H:%M:%S'),
                                    'label': result_label,
                                    'index': pred_idx,
                                    'confidence': confidence
                                })
                        
                        # Reset buffer
                        buffer = []
                        speech_started = False
        finally:
            print("Stopping inference thread...")
            self.mic.stop()
            
    def stop(self):
        self.running = False
