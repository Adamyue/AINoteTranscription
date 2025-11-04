#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined GRU + Attention Training Script
Works directly with feature/pitch .npy files
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

from vocano.model.GRU_Attention import GRU_Attention_Model, Simplified_GRU_Attention_Model

class VOCANODataset(Dataset):
    """Dataset for VOCANO feature/pitch data"""
    def __init__(self, features, labels, augment=False):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.augment and self.training:
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
            
        return feature, label

def load_feature_pitch_data(feat_dir="generated/feat", pitch_dir="generated/pitch", window_size=19):
    """
    Load feature and pitch data directly from .npy files
    """
    print("Loading feature and pitch data...")
    
    feat_files = list(Path(feat_dir).glob("*.npy"))
    all_features = []
    all_labels = []
    
    for feat_file in feat_files:
        # Find corresponding pitch file
        base_name = feat_file.stem.replace("_feat", "_pitch")
        pitch_file = Path(pitch_dir) / f"{base_name}.npy"
        
        if pitch_file.exists():
            print(f"Processing: {feat_file.name}")
            
            # Load data
            features = np.load(feat_file)  # Shape: (1566, time_frames)
            pitch = np.load(pitch_file)    # Shape: (time_frames,)
            
            # Reshape features to (9, 174, time_frames) - more realistic than 522
            features_reshaped = features.reshape(9, 174, -1)
            
            # Pad to 522 if needed (to match model expectations)
            if features_reshaped.shape[1] < 522:
                padding = np.zeros((9, 522 - features_reshaped.shape[1], features_reshaped.shape[2]))
                features_reshaped = np.concatenate([features_reshaped, padding], axis=1)
            
            # Create sliding windows
            time_frames = features_reshaped.shape[2]
            num_windows = time_frames - window_size + 1
            
            # Create windows
            for i in range(num_windows):
                # Extract feature window
                feature_window = features_reshaped[:, :, i:i+window_size]
                
                # Create labels from pitch (binary: 0=no pitch, 1=pitch)
                pitch_window = pitch[i:i+window_size]
                label_window = (pitch_window > 0).astype(np.float32)
                
                all_features.append(feature_window)
                all_labels.append(label_window)
    
    if not all_features:
        print("No valid data found!")
        return None, None, None, None
    
    # Convert to numpy arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"Total samples created: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Use center frame as target
        center_frame = labels.shape[1] // 2
        target_labels = labels[:, center_frame].long()
        
        loss = criterion(outputs, target_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            # Use center frame as target
            center_frame = labels.shape[1] // 2
            target_labels = labels[:, center_frame].long()
            
            loss = criterion(outputs, target_labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

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
    parser = argparse.ArgumentParser(description='Train GRU + Attention Model with Feature/Pitch Data')
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
    parser.add_argument('--pitch_dir', type=str, default='generated/pitch',
                       help='Directory containing pitch .npy files')
    parser.add_argument('--window_size', type=int, default=19,
                       help='Window size for sliding window')
    
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
        'pitch_dir': args.pitch_dir,
        'window_size': args.window_size
    }
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Load data
    train_features, train_labels, val_features, val_labels = load_feature_pitch_data(
        args.feat_dir, args.pitch_dir, args.window_size
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
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
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