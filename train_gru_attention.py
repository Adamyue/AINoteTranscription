#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined GRU + Attention Training Script
Works with feature .npy files and ground truth CSV files
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import signal
import sys

from vocano.model.GRU_Attention import GRU_Attention_Model, Simplified_GRU_Attention_Model
from vocano.utils.gt_to_frame_labels import create_training_labels

class VOCANODataset(Dataset):
    """Dataset for VOCANO feature data with ground truth labels"""
    def __init__(self, features, labels, augment=False):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Apply data augmentation if enabled
        if self.augment:
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
            
        return feature, label

def load_feature_gt_data(feat_dir="generated/feat", gt_dir="dataset/gt_notes", 
                         window_size=19, frame_time=0.02):
    """
    Load feature data and ground truth labels from CSV files.
    
    Args:
        feat_dir: Directory containing feature .npy files
        gt_dir: Directory containing ground truth CSV files (format: start_time, end_time, frequency)
        window_size: Size of sliding window
        frame_time: Time per frame in seconds (default: 0.02s = 20ms)
    
    Returns:
        train_features, train_labels, val_features, val_labels
        - features: (N, 9, 522, window_size) numpy array
        - labels: (N, 6) numpy array of [s, a, o, ō, f, f̄] for center frame
    """
    print("Loading feature and ground truth data...")
    
    feat_files = list(Path(feat_dir).glob("*.npy"))
    all_features = []
    all_labels = []
    
    gt_path = Path(gt_dir)
    
    for feat_file in feat_files:
        # Extract base name (remove _feat suffix if present)
        base_name = feat_file.stem.replace("_feat", "")
        # Remove _vocals suffix if present (for demucs-separated files)
        if base_name.endswith("_vocals"):
            base_name = base_name[:-7]
        
        # Find corresponding ground truth CSV file
        gt_file = gt_path / f"{base_name}.csv"
        
        if not gt_file.exists():
            print(f"Warning: GT file not found for {feat_file.name}: {gt_file}")
            continue
        
        print(f"Processing: {feat_file.name} -> {gt_file.name}")
        
        # Load feature data
        features = np.load(feat_file)  # Shape: (1566, time_frames) or (9, 174, time_frames)
        
        # Reshape features
        if features.ndim == 2:
            # Flattened format: (1566, time_frames) -> (9, 174, time_frames)
            features_reshaped = features.reshape(9, 174, -1)
        else:
            # Already in correct format: (9, 174, time_frames)
            features_reshaped = features
        
        # Pad to 522 if needed (to match model expectations)
        if features_reshaped.shape[1] < 522:
            padding = np.zeros((9, 522 - features_reshaped.shape[1], features_reshaped.shape[2]))
            features_reshaped = np.concatenate([features_reshaped, padding], axis=1)
        
        # Get number of frames
        num_frames = features_reshaped.shape[2]
        
        # Load ground truth and convert to 6-D frame-level labels
        try:
            frame_labels = create_training_labels(gt_file, num_frames, frame_time)  # Shape: (num_frames, 6)
        except Exception as e:
            print(f"Error loading GT for {gt_file}: {e}")
            continue
        
        # Create sliding windows
        time_frames = features_reshaped.shape[2]
        num_windows = time_frames - window_size + 1
        
        # Create windows
        for i in range(num_windows):
            # Extract feature window
            feature_window = features_reshaped[:, :, i:i+window_size]
            
            # Use center frame's 6-D label
            center_frame = i + window_size // 2
            if center_frame < num_frames:
                label_6d = frame_labels[center_frame]  # Shape: (6,)
                
                all_features.append(feature_window)
                all_labels.append(label_6d)
    
    if not all_features:
        print("No valid data found!")
        return None, None, None, None
    
    # Convert to numpy arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"Total samples created: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Label format: [s, a, o, ō, f, f̄]")
    
    # Split into train/validation (80/20)
    num_samples = len(features)
    split_idx = int(0.8 * num_samples)
    
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    
    return train_features, train_labels, val_features, val_labels

def create_model(model_type, hidden_dim=256, num_layers=2):
    """Create model based on configuration"""
    if model_type == 'gru_attention':
        model = GRU_Attention_Model(
            conv1_in_channel=9,
            num_classes=6,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    elif model_type == 'simplified_gru_attention':
        model = Simplified_GRU_Attention_Model(
            conv1_in_channel=9,
            num_classes=6,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def vocano_multi_task_loss(outputs, targets):
    """
    VOCANO paper's multi-task loss function (Equation 1).
    
    LSEG = BCE(ytri, ŷtri) + BCE(yact, ŷact) + BCE(yon, ŷon) + BCE(yoff, ŷoff)
    
    Where:
    - outputs: (batch, 6) model predictions [s, a, o, ō, f, f̄]
    - targets: (batch, 6) ground truth labels [s, a, o, ō, f, f̄]
    - ŷtri = [s, a, t] where t = max(o, f) - transition state
    - ŷact = [s, a] - activation/silence
    - ŷon = [o, ō] - onset/non-onset
    - ŷoff = [f, f̄] - offset/non-offset
    
    Args:
        outputs: (batch_size, 6) tensor of model predictions
        targets: (batch_size, 6) tensor of ground truth labels
    
    Returns:
        Total multi-task BCE loss
    """
    # Apply sigmoid to get probabilities (model outputs logits)
    probs = torch.sigmoid(outputs)
    
    # Extract components: [s, a, o, ō, f, f̄]
    s_pred, a_pred, o_pred, o_bar_pred, f_pred, f_bar_pred = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3], probs[:, 4], probs[:, 5]
    s_true, a_true, o_true, o_bar_true, f_true, f_bar_true = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]
    
    # Compute transition state: t = max(o, f)
    t_pred = torch.max(o_pred, f_pred)
    t_true = torch.max(o_true, f_true)
    
    # BCE loss for each subspace
    bce = nn.BCELoss(reduction='mean')
    
    # 1. Transition subspace: [s, a, t] (3-D)
    ytri_pred = torch.stack([s_pred, a_pred, t_pred], dim=1)
    ytri_true = torch.stack([s_true, a_true, t_true], dim=1)
    loss_tri = bce(ytri_pred, ytri_true)
    
    # 2. Activation subspace: [s, a] (2-D)
    yact_pred = torch.stack([s_pred, a_pred], dim=1)
    yact_true = torch.stack([s_true, a_true], dim=1)
    loss_act = bce(yact_pred, yact_true)
    
    # 3. Onset subspace: [o, ō] (2-D)
    yon_pred = torch.stack([o_pred, o_bar_pred], dim=1)
    yon_true = torch.stack([o_true, o_bar_true], dim=1)
    loss_on = bce(yon_pred, yon_true)
    
    # 4. Offset subspace: [f, f̄] (2-D)
    yoff_pred = torch.stack([f_pred, f_bar_pred], dim=1)
    yoff_true = torch.stack([f_true, f_bar_true], dim=1)
    loss_off = bce(yoff_pred, yoff_true)
    
    # Total loss (sum of all sub-tasks)
    total_loss = loss_tri + loss_act + loss_on + loss_off
    
    return total_loss

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch using VOCANO multi-task loss"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)  # Shape: (batch, 6)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)  # Shape: (batch, 6)
        
        # Compute VOCANO multi-task loss
        loss = vocano_multi_task_loss(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, device):
    """Validate for one epoch using VOCANO multi-task loss"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)  # Shape: (batch, 6)
            
            outputs = model(features)  # Shape: (batch, 6)
            
            # Compute VOCANO multi-task loss
            loss = vocano_multi_task_loss(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = f"checkpoint/epoch_{epoch+1}_gru_attention.pt"
    Path("checkpoint").mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = f"checkpoint/best_gru_attention_{config['model_type']}.pt"
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved to {best_path}")

def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train GRU + Attention Model with Feature Data and Ground Truth Labels')
    parser.add_argument('--model_type', type=str, default='gru_attention',
                       choices=['gru_attention', 'simplified_gru_attention'],
                       help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='GRU hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GRU layers')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--feat_dir', type=str, default='generated/feat',
                       help='Directory containing feature .npy files')
    parser.add_argument('--window_size', type=int, default=19,
                       help='Window size for sliding window')
    parser.add_argument('--gt_dir', type=str, default='dataset/gt_notes',
                       help='Directory containing ground truth CSV files')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                       help='Device to use: auto (detect GPU), cpu, or cuda/cuda:N')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'feat_dir': args.feat_dir,
        'window_size': args.window_size
    }
    
    # Device selection with GPU support
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠ No GPU detected, using CPU (training will be slower)")
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print("Using CPU (forced)")
    else:
        device = torch.device(args.device)
        if device.type == 'cuda':
            if torch.cuda.is_available():
                print(f"✓ Using GPU: {torch.cuda.get_device_name(device.index if device.index else 0)}")
            else:
                print("⚠ CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            print(f"Using device: {device}")
    
    print(f"Device: {device}")
    print(f"Configuration: {config}")
    
    # Load data
    train_features, train_labels, val_features, val_labels = load_feature_gt_data(
        args.feat_dir, args.gt_dir, args.window_size
    )
    
    if train_features is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create datasets
    train_dataset = VOCANODataset(train_features, train_labels, augment=True)
    val_dataset = VOCANODataset(val_features, val_labels, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = create_model(config['model_type'], config['hidden_dim'], config['num_layers'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer (using AdamW as in VOCANO paper)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    # Note: Loss function is defined in vocano_multi_task_loss() function
    # This implements the VOCANO paper's multi-task BCE loss (Equation 1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    interrupted = False
    
    # Signal handler to save checkpoint on interruption (Ctrl+C)
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n\n⚠ Training interrupted! Saving checkpoint...")
        interrupted = True

        # Save current state before exiting
        interrupt_checkpoint = {
            'epoch': len(train_losses),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config,
            'interrupted': True
        }
        print(interrupt_checkpoint)
        
        interrupt_path = f"checkpoint/interrupted_gru_attention_{config['model_type']}.pt"
        Path("checkpoint").mkdir(exist_ok=True)
        torch.save(interrupt_checkpoint, interrupt_path)

        print(f"✓ Checkpoint saved to {interrupt_path}")
        print("You can resume training by loading this checkpoint.")

        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nStarting training...")
    print("=" * 60)
    print("Note: Press Ctrl+C to interrupt and save checkpoint")
    print("=" * 60)
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation
        val_loss = validate_epoch(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % 5 == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, config, is_best)
        
        print('-' * 50)
    
    # Final results
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Save final model
    final_checkpoint = {
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }
    
    torch.save(final_checkpoint, f"checkpoint/final_gru_attention_{config['model_type']}.pt")
    print(f"Final model saved to checkpoint/final_gru_attention_{config['model_type']}.pt")

if __name__ == "__main__":
    main()