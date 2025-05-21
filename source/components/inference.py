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
        self.vad = VAD(sample_rate=16000, frame_ms=30, mode=2)  
        self.spotter = LiveKeywordSpotter(model_path=model_path)
        self.running = False
        self.daemon = True
        self.results_queue = results_queue
        self.audio_queue = audio_queue
        
    def run(self):
        self.running = True
        self.mic.start()
        buffer = []
        speech_frames = 0
        silence_frames = 0
        min_speech_frames = 20  # At least 600ms of speech required
        silence_threshold = 20  # 600ms of silence to consider speech ended
        
        # State machine states
        WAITING = 0
        POSSIBLE_SPEECH = 1
        SPEECH = 2
        POSSIBLE_END = 3
        
        state = WAITING
        last_error_time = 0
        error_count = 0
        
        try:
            print("Inference thread started, listening...")
            while self.running:
                try:
                    # Get audio frame with timeout to prevent blocking forever
                    frame = self.mic.read()
                    
                    # Send audio frame to UI for visualization if queue exists
                    if self.audio_queue and not self.audio_queue.full():
                        try:
                            self.audio_queue.put_nowait(frame)
                        except queue.Full:
                            # Queue is full, discard oldest frames
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                            self.audio_queue.put_nowait(frame)
                    
                    # Periodically clear buffer if it gets too large
                    if len(buffer) > 200:  # More than 6 seconds of audio
                        buffer = buffer[-100:]  # Keep only the last 3 seconds
                        print("Buffer trimmed to prevent memory issues")
                    
                    is_speech = self.vad.is_speech(frame)
                    
                    # State machine for robust speech detection
                    if state == WAITING:
                        if is_speech:
                            buffer = [frame]
                            speech_frames = 1
                            state = POSSIBLE_SPEECH
                        
                    elif state == POSSIBLE_SPEECH:
                        buffer.append(frame)
                        if is_speech:
                            speech_frames += 1
                            if speech_frames >= 5:  # Need 150ms of consistent speech
                                state = SPEECH
                                print("Speech detected!")
                                if self.results_queue and not self.results_queue.full():
                                    self.results_queue.put({
                                        'timestamp': time.strftime('%H:%M:%S'),
                                        'label': "Speech detected...",
                                        'index': -1
                                    })
                        else:
                            speech_frames = 0
                            if len(buffer) > 10:  # If we've collected enough frames, check if it was noise
                                buffer = []
                            state = WAITING
                    
                    elif state == SPEECH:
                        buffer.append(frame)
                        if is_speech:
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1
                            if silence_frames >= 3:  # Short pause (90ms)
                                state = POSSIBLE_END
                            
                    elif state == POSSIBLE_END:
                        buffer.append(frame)
                        if is_speech:
                            # Was just a pause in speech, not the end
                            state = SPEECH
                            silence_frames = 0
                        else:
                            silence_frames += 1
                            if silence_frames >= silence_threshold:
                                # Confirmed end of speech
                                if len(buffer) >= min_speech_frames:
                                    print(f"Processing utterance: {len(buffer)} frames")
                                    
                                    # Process the audio without excessive silence at the end
                                    speech_buffer = buffer[:-silence_frames+3]
                                    audio = np.concatenate(speech_buffer)
                                    
                                    try:
                                        # Process the audio with timeout protection
                                        pred_idx, pred_label, confidence = self.spotter.predict(audio)
                                        
                                        # Format confidence as percentage with 2 decimal places
                                        confidence_pct = confidence * 100
                                        
                                        # Only add to results if confidence is above threshold
                                        if confidence > 0.3:  # 30% confidence threshold
                                            result_label = f"{pred_label} ({confidence_pct:.2f}%)"
                                            print(f"Predicted: {result_label}")
                                            
                                            if self.results_queue and not self.results_queue.full():
                                                self.results_queue.put({
                                                    'timestamp': time.strftime('%H:%M:%S'),
                                                    'label': result_label,
                                                    'index': pred_idx,
                                                    'confidence': confidence
                                                })
                                    except Exception as e:
                                        print(f"Error during prediction: {e}")
                                
                                # Reset for next utterance
                                buffer = []
                                speech_frames = 0
                                silence_frames = 0
                                state = WAITING
                
                except Exception as e:
                    current_time = time.time()
                    if current_time - last_error_time > 5:  # Reset counter if errors are spaced out
                        error_count = 0
                        
                    error_count += 1
                    last_error_time = current_time
                    
                    print(f"Error in inference loop: {e}")
                    
                    if error_count > 10:  # Too many errors in a short time
                        print("Too many errors, resetting microphone...")
                        try:
                            self.mic.stop()
                            time.sleep(1)
                            self.mic.start()
                            buffer = []
                            state = WAITING
                            error_count = 0
                        except Exception as reset_error:
                            print(f"Error resetting microphone: {reset_error}")
                    
                    # Brief pause to avoid tight error loops
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Critical error in inference thread: {e}")
        finally:
            print("Stopping inference thread...")
            try:
                self.mic.stop()
            except Exception as e:
                print(f"Error stopping microphone: {e}")
            
    def stop(self):
        self.running = False
        # Give thread time to clean up
        time.sleep(0.5)

