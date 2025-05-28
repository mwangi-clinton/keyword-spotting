import argparse
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import jiwer
from tqdm import tqdm

from models import SbuLSTMFusion

class AudioFeatureDataset(Dataset):
    def __init__(self, root_dir,
                 sr=16000,
                 n_fft=2048,
                 hop_percent=0.75,
                 win_length=1600,
                 n_mels=40,
                 n_mfcc=13,
                 max_len=None,
                 exclude_folders=None):
        """
        Dataset for evaluation - returns audio features and ground truth labels
        """
        super().__init__()
        self.root_dir = root_dir
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = int(win_length * (1-hop_percent))
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.exclude_folders = set(exclude_folders or [])

        self._prepare_dataset()

    def _prepare_dataset(self):
        # Find all subfolders (classes), filter excludes
        labels = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
               and d not in self.exclude_folders
        ])
        # Build label→index map and reverse mapping for evaluation
        self.label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
        self.idx_to_label = {idx: lab for lab, idx in self.label_to_idx.items()}

        # Walk filesystem and store file paths with labels
        self.samples = []
        for lab in labels:
            folder = os.path.join(self.root_dir, lab)
            for fn in os.listdir(folder):
                if fn.lower().endswith(('.wav','.mp3')):
                    file_path = os.path.join(folder, fn)
                    self.samples.append((file_path, self.label_to_idx[lab], lab, fn))

        # Sanity check
        N = len(self.label_to_idx)
        self.samples = [s for s in self.samples if 0 <= s[1] < N]
        print(f"[Evaluation Dataset] Found {len(self.samples)} files "
              f"under {N} classes: {labels}")

    def __len__(self):
        return len(self.samples)

    def _extract_features(self, path):
        y, _ = librosa.load(path, sr=self.sr)
        # Mel-spectrogram
        m = librosa.feature.melspectrogram(
            y=y, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels)
        log_mel = librosa.power_to_db(m)
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        # Pad/truncate in time
        if self.max_len is not None:
            T = log_mel.shape[1]
            if T >= self.max_len:
                log_mel = log_mel[:, :self.max_len]
                mfcc = mfcc[:, :self.max_len]
            else:
                pw = self.max_len - T
                log_mel = np.pad(log_mel, ((0,0),(0,pw)), mode='constant')
                mfcc = np.pad(mfcc, ((0,0),(0,pw)), mode='constant')
        return log_mel, mfcc

    def __getitem__(self, idx):
        path, label_idx, label_name, filename = self.samples[idx]
        log_mel, mfcc = self._extract_features(path)
        return {
            'mel': torch.FloatTensor(log_mel).unsqueeze(0),
            'mfcc': torch.FloatTensor(mfcc).unsqueeze(0),
            'label': torch.LongTensor([label_idx]).squeeze(),
            'label_name': label_name,
            'filename': filename,
            'file_path': path
        }

def get_eval_dataloader(root_dir,
                       mel_dim=40,
                       mfcc_dim=13,
                       max_len=20,
                       batch_size=32,
                       num_workers=4,
                       exclude_folders=[],
                       hop_percent=0.75,
                       win_length=1600):
    """
    Returns evaluation dataloader optimized for WER computation
    """
    ds = AudioFeatureDataset(
        root_dir=root_dir,
        n_mels=mel_dim,
        n_mfcc=mfcc_dim,
        max_len=max_len,
        exclude_folders=exclude_folders,
        hop_percent=hop_percent,
        win_length=win_length
    )
    
    num_classes = len(ds.label_to_idx)
    
    eval_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return eval_loader, num_classes, ds.label_to_idx, ds.idx_to_label