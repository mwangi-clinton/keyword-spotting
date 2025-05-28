import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# make sure CUDA errors sync
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataloader import get_dataloaders
from models import SbuLSTMFusion

def restricted_float(x):
    """Restrict float values to be between 0.0 and 1.0"""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LSTM Fusion model for speech data")
    parser.add_argument('-f', type=str, required=True, 
                       help='Path to the data folder (e.g., /path/to/speech-data/train/audio/)')
    parser.add_argument('-win', type=int, required=True, 
                       help='Window length for feature extraction (e.g., 800)')
    parser.add_argument('-hop', type=restricted_float, required=True, 
                       help='Hop percent as float between 0.0 and 1.0 (e.g., 0.5)')
    parser.add_argument('-o', type=str, required=True, 
                       help='Folder to save the trained model (e.g., /path/to/models/)')
    parser.add_argument('-m', type=str, default='ideal_model_repeat.pth',
                       help='Name of the model file (default: ideal_model_repeat.pth)')
    return parser.parse_args()

def main():

    args = parse_args()

    root_dir = args.f
    win_length = args.win
    hop_percent = args.hop
    save_folder = args.o
    model_name = args.m
    
    # Ensuring to that save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Lets validate parsed data
    if not os.path.exists(root_dir):
        raise ValueError(f"Data folder does not exist: {root_dir}")
    
    print(f"Configuration:")
    print(f"  Data folder: {root_dir}")
    print(f"  Window length: {win_length}")
    print(f"  Hop percent: {hop_percent}")
    print(f"  Save folder: {save_folder}")
    print(f"  Model name: {model_name}")
    print("-" * 50)
    
    # Fixed hyper-params
    mel_dim = 40
    mfcc_dim = 13
    max_len = 30
    batch_size = 256
    val_frac = 0.2
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    step_size = 3
    gamma = 0.5
    hidden_dim = 128
    uni_layers = 1
    dropout = 0.3
    num_workers = 8
    exclude_folders = []

    # 1) Data
    train_loader, val_loader, num_classes, label_map = get_dataloaders(
        root_dir=root_dir,
        mel_dim=mel_dim,
        mfcc_dim=mfcc_dim,
        max_len=max_len,
        batch_size=batch_size,
        val_frac=val_frac,
        num_workers=num_workers,
        exclude_folders=exclude_folders,
        hop_percent=hop_percent,
        win_length=win_length
    )
    print(f"Classes ({num_classes}): {list(label_map.keys())}")

    # 2) Model / Optim / Scheduler / Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SbuLSTMFusion(
        n_mels=mel_dim,
        n_mfcc=mfcc_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
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
    model_save_path = os.path.join(save_folder, model_name)
    
    for epoch in range(1, epochs+1):
        # ——— TRAIN ———
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            mel = batch['mel'].unsqueeze(1).to(device)
            mfcc = batch['mfcc'].unsqueeze(1).to(device)
            labels = batch['label'].to(device)

            logits = model(mel, mfcc)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # ——— VALIDATE ———
        model.eval()
        val_loss = 0.0
        val_corr = 0
        val_tot = 0
        with torch.no_grad():
            for batch in val_loader:
                mel = batch['mel'].unsqueeze(1).to(device)
                mfcc = batch['mfcc'].unsqueeze(1).to(device)
                labels = batch['label'].to(device)

                logits = model(mel, mfcc)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)

                preds = logits.argmax(dim=1)
                val_corr += (preds == labels).sum().item()
                val_tot += labels.size(0)

        val_loss /= val_tot
        val_acc = 100 * val_corr / val_tot

        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        # save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to: {model_save_path}")
            best_val_acc = val_acc

    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {model_save_path}")

if __name__ == "__main__":
    main()
