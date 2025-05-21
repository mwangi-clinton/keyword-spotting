import queue
import os
import time
import threading
import numpy as np

from source.components.data_ingestion import MicrophoneStreamer
from source.components.vad import VAD
from source.components.predict import LiveKeywordSpotter

class InferenceThread(threading.Thread):
    def __init__(self, model_path='/home/clinton-mwangi/Desktop/msc-telecom/1.2/Speech/kws/keyword-spotting/source/components/model2.pth', results_queue=None, audio_queue=None):
        super().__init__()
        self.mic = MicrophoneStreamer(rate=16000, frame_ms=30)
        self.vad = VAD(sample_rate=16000, frame_ms=30, mode=3)  
        self.spotter = LiveKeywordSpotter(model_path=model_path)
        self.running = False
        self.daemon = True
        self.results_queue = results_queue
        self.audio_queue = audio_queue
        
    def run(self):
        self.running = True
        self.mic.start()
        buffer = []
        speech_started = False
        silence_frames = 0
        min_speech_frames = 10  # At least 300ms of speech required
        silence_threshold = 10  # 300ms of silence to consider speech ended
        
        try:
            print("Inference thread started, listening...")
            while self.running:
                frame = self.mic.read()  # 30 ms samples, numpy float32
                
                # Send audio frame to UI for visualization if queue exists
                if self.audio_queue:
                    self.audio_queue.put(frame)
                
                is_speech = self.vad.is_speech(frame)
                
                if is_speech:
                    if not speech_started:
                        print("Speech detected!")
                        if self.results_queue:
                            self.results_queue.put({
                                'timestamp': time.strftime('%H:%M:%S'),
                                'label': "Speech detected...",
                                'index': -1
                            })
                    
                    buffer.append(frame)
                    speech_started = True
                    silence_frames = 0
                else:
                    if speech_started:
                        silence_frames += 1
                        buffer.append(frame)  # Keep some silence frames for context
                        
                        # End of utterance after sufficient silence
                        if silence_frames >= silence_threshold:
                            if len(buffer) > min_speech_frames:
                                print(f"Processing utterance: {len(buffer)} frames")
                                
                                # Extract the audio without excessive silence at the end
                                speech_buffer = buffer[:-silence_threshold+2]  # Keep a bit of silence
                                audio = np.concatenate(speech_buffer)
                                
                                # Process the audio
                                pred_idx, pred_label, confidence = self.spotter.predict(audio)
                                
                                # Format confidence as percentage with 2 decimal places
                                confidence_pct = confidence * 100
                                
                                # Only add to results if confidence is above threshold
                                if confidence > 0.3:  # 30% confidence threshold
                                    result_label = f"{pred_label} ({confidence_pct:.2f}%)"
                                    print(f"Predicted: {result_label}")
                                    
                                    if self.results_queue:
                                        self.results_queue.put({
                                            'timestamp': time.strftime('%H:%M:%S'),
                                            'label': result_label,
                                            'index': pred_idx,
                                            'confidence': confidence
                                        })
                            
                            # Reset buffer and state
                            buffer = []
                            speech_started = False
                            silence_frames = 0
        finally:
            print("Stopping inference thread...")
            self.mic.stop()
            
    def stop(self):
        self.running = False
