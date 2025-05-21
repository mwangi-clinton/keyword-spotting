import torch
import torch.nn.functional as F
import torch.nn as nn


# class MelCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(MelCNN, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),  
#             nn.ReLU(),
#             nn.MaxPool2d(2),                            

#             nn.Conv2d(16, 32, kernel_size=3, padding=1), 
#             nn.ReLU(),
#             nn.MaxPool2d(2),                             

#         self.classifier = nn.Sequential(
#             nn.Flatten(),                               
#             nn.Linear(32 * 10 * 5, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.classifier(x)
#         return x

class SbuLSTMClassifier(nn.Module):
    def __init__(self,
                 n_mels: int            = 40,
                 hidden_dim: int        = 128,
                 uni_layers: int        = 1,
                 num_classes: int       = 10,
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
        last = x[:, -1, :]             

        # 4) Classify
        logits = self.classifier(last)  
        return logits

class SbuLSTMFusion(nn.Module):
    def __init__(self, 
                 n_mels=40, 
                 n_mfcc=13, 
                 hidden_dim=256, 
                 num_classes=31, 
                 dropout=0.3):
        super().__init__()
        
        # Conv front-end for mel spectrograms
        self.conv_mel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),  # Pool in frequency domain only
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        )
        
        # Conv front-end for MFCC
        self.conv_mfcc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,1)),  # No pooling for MFCC (already low-dim)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Calculate output dimensions after convolutions
        mel_conv_out = n_mels // 4  # After two /2 max-pooling operations
        mfcc_conv_out = n_mfcc     # No reduction in MFCC dimension
        
        # Deeper BiLSTMs for mel features
        self.bdlstm_mel = nn.LSTM(
            input_size=64 * mel_conv_out,  # 64 channels × reduced frequency bins
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Deeper BiLSTMs for MFCC features
        self.bdlstm_mfcc = nn.LSTM(
            input_size=64 * mfcc_conv_out,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism for temporal pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=4*hidden_dim,  # 2*hidden_dim from each bidirectional LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for attention input
        self.layer_norm = nn.LayerNorm(4*hidden_dim)
        
        # Final classifier with improved capacity
        self.classifier = nn.Sequential(
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, mel, mfcc):
        # mel: (B,1,n_mels,T), mfcc: (B,1,n_mfcc,T)
        B, _, _, T = mel.shape
        
        # Apply convolutional front-ends
        mel_conv = self.conv_mel(mel)  # (B,64,n_mels/4,T)
        mfcc_conv = self.conv_mfcc(mfcc)  # (B,64,n_mfcc,T)
        
        # Reshape for LSTM: (B,T,C*F)
        mel_conv = mel_conv.permute(0, 3, 1, 2)  # (B,T,64,n_mels/4)
        mel_conv = mel_conv.reshape(B, T, -1)    # (B,T,64*n_mels/4)
        
        mfcc_conv = mfcc_conv.permute(0, 3, 1, 2)  # (B,T,64,n_mfcc)
        mfcc_conv = mfcc_conv.reshape(B, T, -1)    # (B,T,64*n_mfcc)
        
        # Apply bidirectional LSTMs
        mel_feats, _ = self.bdlstm_mel(mel_conv)   # (B,T,2*hidden_dim)
        mfcc_feats, _ = self.bdlstm_mfcc(mfcc_conv)  # (B,T,2*hidden_dim)
        
        # Concatenate features from both streams
        fused_feats = torch.cat([mel_feats, mfcc_feats], dim=2)  # (B,T,4*hidden_dim)
        
        # Apply layer normalization
        fused_feats = self.layer_norm(fused_feats)
        
        # Self-attention for temporal context
        attn_out, _ = self.attention(fused_feats, fused_feats, fused_feats)
        
        # Residual connection
        fused_feats = fused_feats + attn_out
        
        # Global attention-weighted pooling
        # Create a learnable query vector for attention-based pooling
        query = torch.mean(fused_feats, dim=1, keepdim=True)  # (B,1,4*hidden_dim)
        
        # Calculate attention scores
        attn_scores = torch.bmm(query, fused_feats.transpose(1, 2))  # (B,1,T)
        attn_weights = F.softmax(attn_scores, dim=2)
        
        # Apply attention weights to get context vector
        context = torch.bmm(attn_weights, fused_feats)  # (B,1,4*hidden_dim)
        context = context.squeeze(1)  # (B,4*hidden_dim)
        
        # Final classification
        logits = self.classifier(context)  # (B,num_classes)
        
        return logits