#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert CSV pitch contour to MIDI file
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from vocano.utils.est2midi import Est2MIDI


def pitch_contour_to_notes(frame_times, pitches, min_note_duration=0.1):
    """
    Convert frame-level pitch contour to note-level events.
    
    Args:
        frame_times: Array of frame times (seconds)
        pitches: Array of pitch values (Hz), 0 = silence
        min_note_duration: Minimum note duration in seconds (default 0.1s)
    
    Returns:
        notes: Array of shape (N, 3) with [onset, offset, pitch]
    """
    notes = []
    
    # Find note segments
    i = 0
    while i < len(pitches):
        # Skip silence
        if pitches[i] == 0.0:
            i += 1
            continue
        
        # Found note onset
        onset_idx = i
        onset_time = frame_times[onset_idx]
        
        # Find the end of this note (when pitch changes significantly or goes to 0)
        current_pitch = pitches[i]
        note_pitches = [current_pitch]
        
        i += 1
        while i < len(pitches):
            if pitches[i] == 0.0:
                # Note ended (silence)
                break
            
            # Check if pitch changed significantly (more than 1 semitone)
            pitch_diff_cents = 1200 * np.log2(pitches[i] / current_pitch) if current_pitch > 0 else 0
            if abs(pitch_diff_cents) > 100:  # More than 1 semitone (100 cents)
                # New note started
                break
            
            note_pitches.append(pitches[i])
            i += 1
        
        # Note offset
        offset_idx = i - 1
        offset_time = frame_times[offset_idx]
        
        # Calculate average pitch for this note
        avg_pitch = np.median(note_pitches)
        
        # Check minimum duration
        duration = offset_time - onset_time
        if duration >= min_note_duration and avg_pitch > 0:
            notes.append([onset_time, offset_time, avg_pitch])
    
    return np.array(notes)


def csv_to_midi(csv_path, output_midi_path, min_note_duration=0.1):
    """
    Convert CSV pitch contour file to MIDI.
    
    Args:
        csv_path: Path to CSV file with columns [frame_index, frame_time, pitch]
        output_midi_path: Path to save MIDI file
        min_note_duration: Minimum note duration in seconds
    
    Returns:
        stats: Dictionary with conversion statistics
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    frame_times = df['frame_time'].values
    pitches = df['pitch'].values
    
    print(f"Total frames: {len(frame_times)}")
    print(f"Duration: {frame_times[-1]:.2f} seconds")
    
    # Count non-zero pitches
    voiced_frames = np.sum(pitches > 0)
    print(f"Voiced frames: {voiced_frames} ({voiced_frames/len(pitches)*100:.1f}%)")
    
    # Convert to notes
    print(f"\nConverting pitch contour to notes...")
    print(f"Min note duration: {min_note_duration}s")
    notes = pitch_contour_to_notes(frame_times, pitches, min_note_duration)
    
    print(f"Extracted {len(notes)} notes")
    
    if len(notes) == 0:
        print("⚠️  No notes found in the pitch contour!")
        return {'notes': 0, 'duration': frame_times[-1]}
    
    # Show note statistics
    durations = notes[:, 1] - notes[:, 0]
    print(f"\nNote Statistics:")
    print(f"  - Average duration: {np.mean(durations):.3f}s")
    print(f"  - Min pitch: {np.min(notes[:, 2]):.1f} Hz")
    print(f"  - Max pitch: {np.max(notes[:, 2]):.1f} Hz")
    print(f"  - Pitch range: {np.max(notes[:, 2]) - np.min(notes[:, 2]):.1f} Hz")
    
    # Convert to MIDI
    print(f"\nConverting to MIDI...")
    midi_object = Est2MIDI(notes)
    
    # Save MIDI
    midi_object.write(output_midi_path)
    print(f"✅ MIDI file saved to: {output_midi_path}")
    
    stats = {
        'total_frames': len(frame_times),
        'voiced_frames': int(voiced_frames),
        'duration': float(frame_times[-1]),
        'notes': len(notes),
        'avg_note_duration': float(np.mean(durations)),
        'pitch_range_hz': float(np.max(notes[:, 2]) - np.min(notes[:, 2]))
    }
    
    return stats


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python csv2midi.py <input.csv> [output.mid] [min_duration]")
        print("\nExample:")
        print("  python csv2midi.py ../generated/pitch/MM3_Ex4_librosa_pitch.csv")
        print("  python csv2midi.py ../generated/pitch/MM3_Ex4_librosa_pitch.csv output.mid")
        print("  python csv2midi.py ../generated/pitch/MM3_Ex4_librosa_pitch.csv output.mid 0.05")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Default output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Generate output path from input
        import os
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{base_name}_converted.mid"
    
    # Min note duration
    min_duration = 0.1
    if len(sys.argv) >= 4:
        min_duration = float(sys.argv[3])
    
    print("="*60)
    print("CSV to MIDI Converter")
    print("="*60)
    
    # Convert
    stats = csv_to_midi(csv_path, output_path, min_duration)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"Notes:  {stats['notes']}")
    print(f"Duration: {stats['duration']:.2f}s")
    print("="*60)


if __name__ == '__main__':
    main()

