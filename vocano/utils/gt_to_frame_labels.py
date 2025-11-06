# -*- coding: utf-8 -*-
"""
Convert note-level ground truth annotations to frame-level 6-D labels for training.

Based on VOCANO paper format:
- Output: [s, a, o, ō, f, f̄] where:
  - s: silence probability
  - a: activation (note on) probability  
  - o: onset probability
  - ō: non-onset probability (1 - o)
  - f: offset probability
  - f̄: non-offset probability (1 - f)
"""

import numpy as np
import csv
from pathlib import Path


def notes_to_frame_labels(gt_notes, num_frames, frame_time=0.02):
    """
    Convert note-level annotations to frame-level 6-D labels.
    
    IMPORTANT: This conversion matches the time encoding used in Smooth_sdt6_modified:
    - Frame i corresponds to time: i * frame_time + frame_time/2
    - For frame_time=0.02: frame i → time = 0.02*i + 0.01
    
    This ensures training labels align with how model outputs are decoded during evaluation.
    
    Args:
        gt_notes: (N, 3) array of [start_time, end_time, frequency]
        num_frames: Total number of frames in the audio
        frame_time: Time per frame in seconds (default: 0.02s = 20ms)
    
    Returns:
        labels: (num_frames, 6) array of [s, a, o, ō, f, f̄] probabilities
    """
    labels = np.zeros((num_frames, 6), dtype=np.float32)
    
    # Initialize all frames as silence (s=1, a=0) by default
    labels[:, 0] = 1.0  # s (silence)
    labels[:, 1] = 0.0  # a (activation)
    labels[:, 3] = 1.0  # ō (non-onset)
    labels[:, 5] = 1.0  # f̄ (non-offset)
    
    # Sort notes by start time to handle overlapping notes correctly
    sorted_indices = np.argsort(gt_notes[:, 0])
    
    for idx in sorted_indices:
        start_time, end_time, freq = gt_notes[idx]
        
        # Convert times to frame indices
        # Frame i represents time: i * 0.02 + 0.01
        # So: frame = (time - 0.01) / 0.02
        start_frame = int(np.round((start_time - 0.01) / frame_time))
        end_frame = int(np.round((end_time - 0.01) / frame_time))
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))
        
        # Ensure end_frame >= start_frame
        if end_frame < start_frame:
            end_frame = start_frame
        
        # Mark onset frame (first frame of note)
        # Note: Onset can co-occur with activation in the same frame
        labels[start_frame, 2] = 1.0  # o (onset)
        labels[start_frame, 3] = 0.0  # ō (non-onset)
        
        # Mark offset frame (last frame of note)
        # The offset frame is the frame containing the end_time
        # Note: If note is very short (1 frame), offset = onset frame
        if end_frame > start_frame:
            labels[end_frame, 4] = 1.0  # f (offset)
            labels[end_frame, 5] = 0.0  # f̄ (non-offset)
        else:
            # Very short note: offset same as onset
            labels[start_frame, 4] = 1.0
            labels[start_frame, 5] = 0.0
        
        # Mark activation frames (all frames within note, including onset/offset)
        # Note: Activation includes all frames from start_frame to end_frame (inclusive)
        for frame_idx in range(start_frame, end_frame + 1):
            if 0 <= frame_idx < num_frames:
                labels[frame_idx, 1] = 1.0  # a (activation)
                labels[frame_idx, 0] = 0.0  # s (silence)
    
    return labels


def load_gt_csv(gt_csv_path):
    """
    Load ground truth CSV file.
    
    Args:
        gt_csv_path: Path to CSV file with format: start_time,end_time,frequency
    
    Returns:
        gt_notes: (N, 3) array of [start_time, end_time, frequency]
    """
    gt_notes = []
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                start_time = float(row[0])
                end_time = float(row[1])
                frequency = float(row[2])
                gt_notes.append([start_time, end_time, frequency])
    
    return np.array(gt_notes, dtype=np.float32)


def create_training_labels(gt_csv_path, num_frames, frame_time=0.02):
    """
    Convenience function to load GT and convert to frame labels.
    
    Args:
        gt_csv_path: Path to ground truth CSV
        num_frames: Number of frames in the audio
        frame_time: Time per frame (default: 0.02s)
    
    Returns:
        labels: (num_frames, 6) array of frame-level labels
    """
    gt_notes = load_gt_csv(gt_csv_path)
    labels = notes_to_frame_labels(gt_notes, num_frames, frame_time)
    return labels


if __name__ == "__main__":
    # Example usage
    gt_path = Path("dataset/gt_notes/A_Day_To_Remember-Fast_Forward_To_2012.csv")
    num_frames = 1000  # Example: adjust based on audio duration
    
    labels = create_training_labels(gt_path, num_frames)
    print(f"Labels shape: {labels.shape}")
    print(f"First 5 frames:\n{labels[:5]}")
    print(f"Onset frames: {np.where(labels[:, 2] > 0.5)[0]}")
    print(f"Offset frames: {np.where(labels[:, 4] > 0.5)[0]}")

