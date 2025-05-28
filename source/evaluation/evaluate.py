import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import jiwer

from dataloader import get_eval_dataloader
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
    parser = argparse.ArgumentParser(description="Evaluate LSTM Fusion model for speech data")
    parser.add_argument('--data-folder', type=str, required=True, 
                       help='Path to evaluation data folder')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to the trained model file (.pth)')
    parser.add_argument('--win-length', type=int, default=1600,
                       help='Window length for feature extraction (default: 1600)')
    parser.add_argument('--hop-percent', type=restricted_float, default=0.75,
                       help='Hop percent as float between 0.0 and 1.0 (default: 0.75)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--max-len', type=int, default=20, 
                       help='Max length for feature padding/truncation (default: 20)')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of workers for DataLoader (default: 4)')
    parser.add_argument('--exclude-folders', nargs='*', default=['.ipynb_checkpoints'], 
                       help='Folders to exclude from dataset (default: [.ipynb_checkpoints])')
    parser.add_argument('--mel-dim', type=int, default=40,
                       help='Number of mel frequency bins (default: 40)')
    parser.add_argument('--mfcc-dim', type=int, default=13,
                       help='Number of MFCC coefficients (default: 13)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension of LSTM (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results per class')
    return parser.parse_args()

def evaluate_with_wer(model, eval_loader, idx_to_label, device='cpu', verbose=False):
    """
    Evaluation function that computes WER and accuracy
    """
    model.eval()
    predictions = []
    ground_truths = []
    correct = 0
    total = 0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            mel_features = batch['mel'].to(device)
            mfcc_features = batch['mfcc'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(mel_features, mfcc_features)
            pred_indices = torch.argmax(logits, dim=1)
            
            # Calculate accuracy
            correct += (pred_indices == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy tracking
            for i in range(len(pred_indices)):
                label_name = batch['label_name'][i]
                pred_correct = (pred_indices[i] == labels[i]).item()
                
                if label_name not in class_correct:
                    class_correct[label_name] = 0
                    class_total[label_name] = 0
                
                class_correct[label_name] += pred_correct
                class_total[label_name] += 1
            
            # Convert predictions to words for WER
            for pred_idx in pred_indices.cpu().numpy():
                predictions.append(idx_to_label[pred_idx])
            
            # Get ground truth words
            for label_name in batch['label_name']:
                ground_truths.append(label_name)
    
    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for class_name in class_total:
        class_accuracies[class_name] = 100.0 * class_correct[class_name] / class_total[class_name]
    
    return predictions, ground_truths, accuracy, class_accuracies

def main():
 
    args = parse_args()
    
    # validating inputs
    if not os.path.exists(args.data_folder):
        raise ValueError(f"Data folder does not exist: {args.data_folder}")
    
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model file does not exist: {args.model_path}")
    
    print(f"Evaluation Configuration:")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Model path: {args.model_path}")
    print(f"  Window length: {args.win_length}")
    print(f"  Hop percent: {args.hop_percent}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_len}")
    print("-" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load evaluation data
    eval_loader, num_classes, label_to_idx, idx_to_label = get_eval_dataloader(
        root_dir=args.data_folder,
        mel_dim=args.mel_dim,
        mfcc_dim=args.mfcc_dim,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        exclude_folders=args.exclude_folders,
        hop_percent=args.hop_percent,
        win_length=args.win_length
    )
    
    print(f"Found {num_classes} classes: {list(label_to_idx.keys())}")
    
    # Load model
    model = SbuLSTMFusion(
        n_mels=args.mel_dim,
        n_mfcc=args.mfcc_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from: {args.model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    predictions, ground_truths, accuracy, class_accuracies = evaluate_with_wer(
        model, eval_loader, idx_to_label, device, args.verbose
    )
    
    # Compute WER
    wer = jiwer.wer(ground_truths, predictions)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Word Recognition Rate: {(1-wer)*100:.2f}%")
    
    if args.verbose:
        print("\nPer-class Accuracy:")
        print("-" * 30)
        for class_name, acc in sorted(class_accuracies.items()):
            print(f"  {class_name:15s}: {acc:6.2f}%")
    
    print(f"\nTotal samples evaluated: {len(ground_truths)}")
    print("="*50)

if __name__ == '__main__':
    main()
