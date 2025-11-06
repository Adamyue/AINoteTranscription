#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch feature extraction script for VOCANO
Extracts CFP features from all audio files in dataset/wav_audio_demucsed
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from vocano.core import SingingVoiceTranscription

def extract_features_batch(wav_dir="dataset/wav_audio_demucsed", 
                          feat_dir="generated/feat",
                          pitch_dir="generated/pitch",
                          device="auto",
                          use_cp=True,
                          batch_size=64,
                          num_workers=0,
                          pin_memory=True):
    """
    Extract features from all WAV files in the specified directory.
    
    Args:
        wav_dir: Directory containing WAV files
        feat_dir: Directory to save feature files
        pitch_dir: Directory to save pitch files
        device: Device to use ('auto', 'cpu', or 'cuda:N')
        use_cp: Use CuPy for GPU acceleration
        batch_size: Batch size for feature extraction
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
    """
    wav_path = Path(wav_dir)
    feat_path = Path(feat_dir)
    pitch_path = Path(pitch_dir)
    
    # Create output directories
    feat_path.mkdir(parents=True, exist_ok=True)
    pitch_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(wav_path.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Features will be saved to: {feat_path}")
    print(f"Pitch files will be saved to: {pitch_path}")
    print("=" * 60)
    
    # Process each file
    processed = 0
    skipped = 0
    errors = 0
    
    for wav_file in tqdm(wav_files, desc="Extracting features"):
        # Get base name (remove .wav extension and _vocals suffix if present)
        base_name = wav_file.stem
        if base_name.endswith("_vocals"):
            base_name = base_name[:-7]  # Remove _vocals suffix
        
        print(f"Processing: {base_name}")

        # Check if feature already exists
        feat_file = feat_path / f"{base_name}_feat.npy"
        if feat_file.exists():
            skipped += 1
            continue
        
        try:
            # Handle device selection for Windows
            actual_device = device
            if device == "auto":
                if torch.cuda.is_available():
                    actual_device = "0"  # Use GPU 0, as expected by core.py
                else:
                    actual_device = "cpu"
            
            # Create args namespace
            args = argparse.Namespace()
            args.name = base_name
            args.feat_dir = feat_path
            args.pitch_dir = pitch_path
            args.midi_dir = Path("generated/midi")
            args.output_wav_dir = Path("generated/wav")
            args.wavfile_dir = wav_file
            args.pitch_gt_dir = Path("groundtruth/pitch.npy")
            args.checkpoint_file = "./checkpoint/model.pt"
            args.save_extracted = True  # Save extracted features
            args.use_pre_extracted = False
            args.use_groundtruth = False
            args.device = actual_device
            args.use_cp = use_cp
            args.batch_size = batch_size
            args.num_workers = num_workers
            args.pin_memory = pin_memory
            args.use_amp = False
            args.model_type = "pyramidnet"  # Not used for preprocessing, but needed
            
            # Create solver and extract features
            solver = SingingVoiceTranscription(args)
            solver.data_preprocessing()
            
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {wav_file.name}: {e}")
            errors += 1
            continue
    
    print("\n" + "=" * 60)
    print(f"Feature extraction completed!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(wav_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch feature extraction for VOCANO")
    parser.add_argument("--wav_dir", type=str, default="dataset/wav_audio_demucsed",
                       help="Directory containing WAV files")
    parser.add_argument("--feat_dir", type=str, default="generated/feat",
                       help="Directory to save feature files")
    parser.add_argument("--pitch_dir", type=str, default="generated/pitch",
                       help="Directory to save pitch files")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                       help="Device to use")
    parser.add_argument("--use_cp", action="store_true", default=True,
                       help="Use CuPy for GPU acceleration")
    parser.add_argument("--no_cp", dest="use_cp", action="store_false",
                       help="Disable CuPy")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers for data loading")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                       help="Pin memory for faster GPU transfer")
    
    args = parser.parse_args()
    
    extract_features_batch(
        wav_dir=args.wav_dir,
        feat_dir=args.feat_dir,
        pitch_dir=args.pitch_dir,
        device=args.device,
        use_cp=args.use_cp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

