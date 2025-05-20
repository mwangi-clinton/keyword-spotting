import torch
import torch.nn.functional as F
import torch.nn as nn


class MelCNN(nn.Module):
    def __init__(self, num_classes):
        super(MelCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 1, 40, 20) → (B, 16, 40, 20)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (B, 16, 20, 10)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (B, 32, 20, 10)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (B, 32, 10, 5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (B, 32*10*5)
            nn.Linear(32 * 10 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

class SbuLSTMClassifier(nn.Module):
    def __init__(self,
                 n_mels: int            = 40,
                 hidden_dim: int        = 128,
                 uni_layers: int        = 1,
                 num_classes: int       = 30,
                 dropout: float         = 0.3):
        super(SbuLSTMClassifier, self).__init__()
        # 1) Bidirectional LSTM feature extractor
        self.bdlstm = nn.LSTM(
            input_size   = n_mels,
            hidden_size  = hidden_dim,
            num_layers   = 1,
            batch_first  = True,
            bidirectional= True
        )
        # 2) Unidirectional LSTM for forward‐only refinement
        self.lstm = nn.LSTM(
            input_size  = 2 * hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = uni_layers,
            batch_first = True,
            bidirectional = False
        )
        # 3) MLP classifier on the last time step
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        B, C, F, T = x.shape
        # remove channel dim → (B, F, T), then time‐first → (B, T, F)
        x = x.view(B, F, T).permute(0, 2, 1)

        # 1) Bidirectional LSTM → (B, T, 2*hidden_dim)
        x, _ = self.bdlstm(x)

        # 2) Unidirectional LSTM → (B, T, hidden_dim)
        x, _ = self.lstm(x)

        # 3) Take last time step features
        last = x[:, -1, :]              # (B, hidden_dim)

        # 4) Classify
        logits = self.classifier(last)  # (B, num_classes)
        return logits

class SbuLSTMFusion(nn.Module):
    def __init__(self,
                 n_mels: int       = 40,
                 n_mfcc: int       = 13,
                 hidden_dim: int   = 128,
                 uni_layers: int   = 1,
                 num_classes: int  = 30,
                 dropout: float    = 0.3):
        super().__init__()
        # Bi-LSTM on Mel
        self.bdlstm_mel = nn.LSTM(
            input_size    = n_mels,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = True
        )
        # Bi-LSTM on MFCC
        self.bdlstm_mfcc = nn.LSTM(
            input_size    = n_mfcc,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = True
        )
        # Unidirectional LSTM after fusion
        fused_dim = 2*hidden_dim*2   # mel(2h) + mfcc(2h)
        self.lstm = nn.LSTM(
            input_size  = fused_dim,
            hidden_size = hidden_dim,
            num_layers  = uni_layers,
            batch_first = True
        )
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        # mel: (B,1,n_mels,T), mfcc: (B,1,n_mfcc,T)
        B,_,F_mel,T = mel.shape
        _,_,F_mfcc,_= mfcc.shape

        # → (B, T, F)
        mel = mel.view(B, F_mel, T).permute(0,2,1)
        mf  = mfcc.view(B, F_mfcc, T).permute(0,2,1)

        # Bi-LSTM branches
        mel_feats, _ = self.bdlstm_mel(mel)   # (B,T,2*hidden)
        mf_feats,  _ = self.bdlstm_mfcc(mf)  # (B,T,2*hidden)

        # fuse
        x = torch.cat([mel_feats, mf_feats], dim=2)  # (B,T,4*hidden)

        # uni-LSTM
        x, _ = self.lstm(x)                          # (B,T,hidden)

        # classify last timestep
        logits = self.classifier(x[:, -1, :])        # (B,num_classes)
        return logits

