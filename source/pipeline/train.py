import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# make sure CUDA errors sync
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataloader import get_dataloaders
from model      import SbuLSTMFusion

def main():
    # hyper‐params
    root_dir      = '/capstor/scratch/cscs/ckuya/speech-data/train/audio/'
    mel_dim       = 40
    mfcc_dim      = 13
    max_len       = 20
    batch_size    = 256
    val_frac      = 0.2
    epochs        = 10
    lr            = 1e-3
    weight_decay  = 1e-5
    step_size     = 3
    gamma         = 0.5
    hidden_dim    = 128
    uni_layers    = 1
    dropout       = 0.3
    num_workers   = 4
    exclude_folders = ['_background_noise_']

    # 1) Data
    train_loader, val_loader, num_classes, label_map = get_dataloaders(
        root_dir        = root_dir,
        mel_dim         = mel_dim,
        mfcc_dim        = mfcc_dim,
        max_len         = max_len,
        batch_size      = batch_size,
        val_frac        = val_frac,
        num_workers     = num_workers,
        exclude_folders = exclude_folders
    )
    print(f"Classes ({num_classes}): {list(label_map.keys())}")

    # 2) Model / Optim / Scheduler / Loss
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = SbuLSTMFusion(
        n_mels     = mel_dim,
        n_mfcc     = mfcc_dim,
        hidden_dim = hidden_dim,
        uni_layers = uni_layers,
        num_classes= num_classes,
        dropout    = dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # 3) Train + Validate
    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        # ——— TRAIN ———
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            mel   = batch['mel'].unsqueeze(1).to(device)
            mfcc  = batch['mfcc'].unsqueeze(1).to(device)
            labels= batch['label'].to(device)

            logits = model(mel, mfcc)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_loss = running_loss/total
        train_acc  = 100*correct/total

        # ——— VALIDATE ———
        model.eval()
        val_loss = 0.0
        val_corr = 0
        val_tot  = 0
        with torch.no_grad():
            for batch in val_loader:
                mel   = batch['mel'].unsqueeze(1).to(device)
                mfcc  = batch['mfcc'].unsqueeze(1).to(device)
                labels= batch['label'].to(device)

                logits = model(mel, mfcc)
                loss   = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)

                preds = logits.argmax(dim=1)
                val_corr += (preds==labels).sum().item()
                val_tot  += labels.size(0)

        val_loss /= val_tot
        val_acc   = 100*val_corr/val_tot

        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        # save best
        if val_acc>best_val_acc:
            torch.save(model.state_dict(),"best_sbulstm_fusion.pth")
            best_val_acc=val_acc

    print(f"Training done → Best Val Acc: {best_val_acc:.2f}%")

if __name__=="__main__":
    main()
