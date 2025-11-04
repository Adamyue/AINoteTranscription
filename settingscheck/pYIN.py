#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio pitch extraction using librosa
Output format: (frame_index, frame_time, pitch)
"""

import librosa
import numpy as np
import pandas as pd
import argparse
import os


def extract_pitch_librosa(audio_path, output_csv=None, hop_length=320, fmin=80.0, fmax=1000.0):
    """
    Extract pitch from audio using librosa's pyin algorithm
    
    Args:
        audio_path: Path to input audio file
        output_csv: Path to output CSV file (optional)
        hop_length: Hop size in samples (default: 320, same as VOCANO)
        fmin: Minimum frequency in Hz (default: 80, same as VOCANO)
        fmax: Maximum frequency in Hz (default: 1000, same as VOCANO)
    
    Returns:
        DataFrame with columns: frame_index, frame_time, pitch
    """
    print(f"Loading audio file: {audio_path}")
    
    # Load audio and resample to 16000 Hz (same as feature_extraction.py)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    print(f"Audio loaded: duration={len(audio)/sr:.2f}s, sample_rate={sr}Hz")
    print(f"Extracting pitch with hop_length={hop_length} samples ({hop_length/sr*1000:.1f}ms)")
    
    # Extract pitch using pyin (probabilistic YIN algorithm)
    # This is one of the best pitch trackers in librosa for monophonic sources
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        frame_length=2048,  # Window size
        center=True,
        pad_mode='constant'
    )
    
    # Calculate frame times
    frame_times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )
    
    # Replace NaN with 0 for unvoiced frames
    f0 = np.nan_to_num(f0, nan=0.0)
    
    # Filter based on voiced probability threshold (0.5)
    # If voiced_probs <= 0.5, consider it as non-vocal and set pitch to 0
    f0[voiced_probs <= 0.5] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame({
        'frame_index': np.arange(len(f0)),
        'frame_time': frame_times,
        'pitch': f0
    })
    
    print(f"Pitch extraction complete: {len(df)} frames")
    print(f"  Voiced frames: {(df['pitch'] > 0).sum()}")
    print(f"  Unvoiced frames: {(df['pitch'] == 0).sum()}")
    
    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract pitch from audio using librosa',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_audio',
        type=str,
        help='Path to input audio file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: <input_name>_librosa_pitch.csv)'
    )
    parser.add_argument(
        '--hop-length',
        type=int,
        default=320,
        help='Hop size in samples (default: 320, matching VOCANO)'
    )
    parser.add_argument(
        '--fmin',
        type=float,
        default=80.0,
        help='Minimum frequency in Hz (default: 80.0)'
    )
    parser.add_argument(
        '--fmax',
        type=float,
        default=1000.0,
        help='Maximum frequency in Hz (default: 1000.0)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_audio):
        print(f"Error: Input file not found: {args.input_audio}")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_audio))[0]
        args.output = f"{base_name}_librosa_pitch.csv"
    
    # Extract pitch
    df = extract_pitch_librosa(
        args.input_audio,
        args.output,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax
    )
    
    # Print statistics
    print("\n=== Pitch Statistics ===")
    voiced_pitches = df[df['pitch'] > 0]['pitch']
    if len(voiced_pitches) > 0:
        print(f"Min pitch: {voiced_pitches.min():.2f} Hz")
        print(f"Max pitch: {voiced_pitches.max():.2f} Hz")
        print(f"Mean pitch: {voiced_pitches.mean():.2f} Hz")
        print(f"Median pitch: {voiced_pitches.median():.2f} Hz")
    else:
        print("No voiced frames detected")


if __name__ == '__main__':
    main()

