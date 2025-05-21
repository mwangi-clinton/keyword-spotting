import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import sounddevice as sd
import webrtcvad
import torch
import librosa
import queue
import numpy as np
# ─── Global state and queues ──────────────────────────────────────────────────
RUNNING = False
THREAD = None
RESULTS_QUEUE = queue.Queue()  # For passing results from inference thread to UI
class LiveKeywordSpotter:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        from source.pipeline.models import SbuLSTMFusion
        # build model & load checkpoint
        self.model = SbuLSTMFusion(
            n_mels=40, n_mfcc=13,
            hidden_dim=128,
            num_classes=31, dropout=0.3
        ).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Target keywords
        self.target_keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        
        # Label mapping
        self.idx2label = {
            0: "_background_noise_", 1: "bed", 2: "bird", 3: "cat", 4: "dog", 5: "down", 
            6: "eight", 7: "five", 8: "four", 9: "go", 10: "happy", 
            11: "house", 12: "left", 13: "marvin", 14: "nine", 15: "no", 
            16: "off", 17: "on", 18: "one", 19: "right", 20: "seven", 
            21: "sheila", 22: "six", 23: "stop", 24: "three", 25: "tree", 
            26: "two", 27: "up", 28: "wow", 29: "yes", 30: "zero"
        }

        # Create a mapping from keywords to indices
        self.target_indices = {label: idx for idx, label in self.idx2label.items() 
                              if label in self.target_keywords}
        print(f"Target keywords: {self.target_keywords}")
        print(f"Target indices: {self.target_indices}")

        # mel/MFCC params
        self.sr = 16000
        self.n_fft = 512  # Reduced from 1024 for faster processing
        self.win_length = 400  # Reduced from 800 for faster processing
        self.hop_length = 160  # Reduced from 400 for faster processing
        self.n_mels = 40
        self.n_mfcc = 13

    def predict(self, audio: np.ndarray):
        """
        audio: 1D float32 np array (mono), length self.sr*N seconds
        We'll extract log-mel + MFCC, pad/truncate to max_len frames,
        then do a model forward pass.
        """
        # 1) compute mel and mfcc
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel_spec)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        # normalize
        log_mel = (log_mel - log_mel.mean())/(log_mel.std()+1e-8)
        mfcc = (mfcc - mfcc.mean())/(mfcc.std()+1e-8)

        # pad/truncate to, say, 40 frames (0.4s)
        max_len = 40
        T = log_mel.shape[1]
        if T < max_len:
            pad = max_len - T
            log_mel = np.pad(log_mel, ((0,0),(0,pad)), 'constant')
            mfcc = np.pad(mfcc, ((0,0),(0,pad)), 'constant')
        else:
            log_mel = log_mel[:, :max_len]
            mfcc = mfcc[:, :max_len]

        # to torch
        mel_tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(mel_tensor, mfcc_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Filter to only include target keywords
            filtered_probs = {idx: probs[idx].item() for idx in self.target_indices.values()}
            
            # Get the highest probability keyword
            if filtered_probs:
                pred_idx = max(filtered_probs, key=filtered_probs.get)
                confidence = filtered_probs[pred_idx]
                
                # Only print predictions with confidence > 0.3
                if confidence > 0.3:
                    print(f"Top prediction: {self.idx2label[pred_idx]} with confidence {confidence:.4f}")
                
                # Return the prediction and confidence regardless of threshold
                pred_label = self.idx2label.get(pred_idx)
                return pred_idx, pred_label, confidence
            
            # Return background noise if no target keyword detected
            return 0, "_background_noise_", 0.0

