#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick script to check available PyTorch devices
"""
import torch

print("=" * 60)
print("PyTorch Device Information")
print("=" * 60)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA is available!")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # List all available GPUs
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        print(f"   Total Memory: {total_memory:.2f} GB")
        
        # Current memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved: {reserved:.2f} GB")
            print(f"   Free: {total_memory - reserved:.2f} GB")
        print()
    
    print("-" * 60)
    print("Recommendation for VOCANO:")
    if torch.cuda.device_count() == 1:
        print(f"✅ Use: -d 0")
        print(f"   (You have 1 GPU)")
    else:
        print(f"✅ Use: -d 0 (or -d 1, -d 2, etc.)")
        print(f"   (You have {torch.cuda.device_count()} GPUs)")
        print(f"   Or use: -d auto (to auto-select best GPU on Linux)")
    print()
    print("For faster inference, add: -use_amp")
    print(f"Example: python -m vocano.transcription -wd audio.wav -d 0 -use_amp")
    
else:
    print("❌ CUDA is NOT available")
    print("   Your system doesn't have NVIDIA GPU support enabled")
    print()
    print("-" * 60)
    print("Recommendation for VOCANO:")
    print(f"✅ Use: -d cpu")
    print(f"   (CPU-only mode)")
    print()
    print("⚠️  Note: AMP mode (-use_amp) won't provide benefits on CPU")
    print(f"Example: python -m vocano.transcription -wd audio.wav -d cpu")

print("=" * 60)

# Check model checkpoint
print()
print("=" * 60)
print("Model Checkpoint Information")
print("=" * 60)

checkpoint_path = '../checkpoint/model.pt'
try:
    import os
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Path: {checkpoint_path}")
        print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Show checkpoint details if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                num_params = len(checkpoint[state_dict_key])
                print(f"   Number of parameter tensors: {num_params}")
        print()
        print("✅ Model checkpoint is ready for inference!")
    else:
        print(f"⚠️  Checkpoint not found at: {checkpoint_path}")
        print(f"   Please ensure the model checkpoint is in the correct location")
        print(f"   Expected: checkpoint/model.pt")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print(f"   Path: {checkpoint_path}")

print("=" * 60)

