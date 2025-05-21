import torch
import librosa
import numpy as np

class LiveKeywordSpotter:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        from source.pipeline.models import SbuLSTMFusion
        # build model & load checkpoint
        self.model = SbuLSTMFusion(
            n_mels=40, n_mfcc=13,
            hidden_dim=128,
            num_classes=10, dropout=0.3
        ).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Target keywords
        self.target_keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        
        # Label mapping
        self.idx2label = {
            0: "down", 1: "go", 2: "left", 3: "no", 4: "off", 
            5: "on", 6: "right", 7: "stop", 8: "up", 9: "yes"
        }

        # Create a mapping from keywords to indices
        self.target_indices = {label: idx for idx, label in self.idx2label.items() 
                              if label in self.target_keywords}
        print(f"Target keywords: {self.target_keywords}")
        print(f"Target indices: {self.target_indices}")

        # mel/MFCC params
        self.sr = 16000
        self.n_fft = 1024  
        self.win_length = 800  
        self.hop_length = 400  
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
            
            # Get the highest probability keyword
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            # Only print predictions with confidence > 0.3
            if confidence > 0.3:
                print(f"Top prediction: {self.idx2label[pred_idx]} with confidence {confidence:.4f}")
            
            # Return the prediction and confidence regardless of threshold
            pred_label = self.idx2label.get(pred_idx)
            return pred_idx, pred_label, confidence
