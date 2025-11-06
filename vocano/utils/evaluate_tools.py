# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:11:59 2020

@author: Austin Hsu
"""

import numpy as np

def find_first_bellow_thres(aSeq):
    activate = False
    first_bellow_frame = 0
    for i in range(len(aSeq)):
        if aSeq[i] > 0.5:
            activate = True
        if activate and aSeq[i] < 0.5:
            first_bellow_frame = i
            break
    return first_bellow_frame

def Smooth_sdt6_modified(predict_sdt, threshold=0.5):
    # predict shape: (time step, 3)
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    sSeq = []
    dSeq = []
    onSeq = []
    offSeq = []
    
    for num in range(predict_sdt.shape[0]):
        if num > 1 and num < predict_sdt.shape[0]-2:
            sSeq.append(predict_sdt[num][0].astype(np.float64))
            dSeq.append(predict_sdt[num][1].astype(np.float64))
            onSeq.append(np.dot(predict_sdt[num-2:num+3, 3], Filter) / 2.5)
            offSeq.append(np.dot(predict_sdt[num-2:num+3, 5], Filter) / 2.5)

        else:
            sSeq.append(predict_sdt[num][0].astype(np.float64))
            dSeq.append(predict_sdt[num][1].astype(np.float64))
            onSeq.append(predict_sdt[num][3])
            offSeq.append(predict_sdt[num][5])   
    
    ##############################
    # Peak strategy
    ##############################
    
    # find peak of transition
    # peak time = frame*0.02+0.01
    onpeaks = []
    if onSeq[0] > onSeq[1] and onSeq[0] > onSeq[2] and onSeq[0] > threshold:
        onpeaks.append(0)
    if onSeq[1] > onSeq[0] and onSeq[1] > onSeq[2] and onSeq[1] > onSeq[3] and onSeq[1] > threshold:
        onpeaks.append(1)
    for num in range(len(onSeq)):
        if num > 1 and num < len(onSeq)-2:
            if onSeq[num] > onSeq[num-1] and onSeq[num] > onSeq[num-2] and onSeq[num] > onSeq[num+1] and onSeq[num] > onSeq[num+2] and onSeq[num] > threshold:
                onpeaks.append(num)

    if onSeq[-1] > onSeq[-2] and onSeq[-1] > onSeq[-3] and onSeq[-1] > threshold:
        onpeaks.append(len(onSeq)-1)
    if onSeq[-2] > onSeq[-1] and onSeq[-2] > onSeq[-3] and onSeq[-2] > onSeq[-4] and onSeq[-2] > threshold:
        onpeaks.append(len(onSeq)-2)


    offpeaks = []
    if offSeq[0] > offSeq[1] and offSeq[0] > offSeq[2] and offSeq[0] > threshold:
        offpeaks.append(0)
    if offSeq[1] > offSeq[0] and offSeq[1] > offSeq[2] and offSeq[1] > offSeq[3] and offSeq[1] > threshold:
        offpeaks.append(1)
    for num in range(len(offSeq)):
        if num > 1 and num < len(offSeq)-2:
            if offSeq[num] > offSeq[num-1] and offSeq[num] > offSeq[num-2] and offSeq[num] > offSeq[num+1] and offSeq[num] > offSeq[num+2] and offSeq[num] > threshold:
                offpeaks.append(num)

    if offSeq[-1] > offSeq[-2] and offSeq[-1] > offSeq[-3] and offSeq[-1] > threshold:
        offpeaks.append(len(offSeq)-1)
    if offSeq[-2] > offSeq[-1] and offSeq[-2] > offSeq[-3] and offSeq[-2] > offSeq[-4] and offSeq[-2] > threshold:
        offpeaks.append(len(offSeq)-2)

    # determine onset/offset by silence, duration
    # intervalSD = [0,1,0,1,...], 0:silence, 1:duration
    if len(onpeaks) == 0 or len(offpeaks) == 0:
        return None
    
    # Clearing out offsets before first onset (since onset is more accurate)
    orig_offpeaks = offpeaks
    offpeaks = [i for i in orig_offpeaks if i>onpeaks[0]]
    
    Tpeaks = onpeaks + offpeaks
    Tpeaks.sort()

    intervalSD = [0]

    for i in range(len(Tpeaks)-1):
        current_sd = 0 if sum(sSeq[Tpeaks[i]:Tpeaks[i+1]]) > sum(dSeq[Tpeaks[i]:Tpeaks[i+1]]) else 1
        intervalSD.append(current_sd)
    intervalSD.append(0)


    MissingT= 0
    AddingT = 0
    est_intervals = []
    t_idx = 0
    while t_idx < len(Tpeaks):
        if t_idx == len(Tpeaks)-1:
            break
        if t_idx == 0 and Tpeaks[t_idx] not in onpeaks:
            t_idx += 1

        if Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in offpeaks:
            if Tpeaks[t_idx] == Tpeaks[t_idx+1]:
                t_idx += 1
                continue
            if Tpeaks[t_idx+1] > Tpeaks[t_idx]+1: 
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*Tpeaks[t_idx+1]+0.01])
            assert(Tpeaks[t_idx] < Tpeaks[t_idx+1])
            t_idx += 2
        elif Tpeaks[t_idx] in onpeaks and Tpeaks[t_idx+1] in onpeaks:
            offset_inserted = find_first_bellow_thres(dSeq[Tpeaks[t_idx]:Tpeaks[t_idx+1]]) + Tpeaks[t_idx]
            if offset_inserted != Tpeaks[t_idx] and offset_inserted > Tpeaks[t_idx]+1:
                est_intervals.append([0.02*Tpeaks[t_idx]+0.01, 0.02*offset_inserted+0.01])
                AddingT += 1
                assert(Tpeaks[t_idx] < offset_inserted)
            else:
                MissingT += 1
            t_idx += 1
        elif Tpeaks[t_idx] in offpeaks:
            t_idx += 1
    
    print("Conflict ratio: ", MissingT/(len(Tpeaks)+AddingT))
    print("Tpeaks: ", Tpeaks)

    # Modify 1
    sSeq_np = np.ndarray(shape=(len(sSeq),), dtype=float, buffer=np.array(sSeq))
    dSeq_np = np.ndarray(shape=(len(dSeq),), dtype=float, buffer=np.array(dSeq))
    onSeq_np = np.ndarray(shape=(len(onSeq),), dtype=float, buffer=np.array(onSeq))
    offSeq_np = np.ndarray(shape=(len(offSeq),), dtype=float, buffer=np.array(offSeq))

    return np.ndarray(shape=(len(est_intervals),2), dtype=float, buffer=np.array(est_intervals)),  sSeq_np, dSeq_np, onSeq_np, offSeq_np, MissingT/(len(Tpeaks)+AddingT)

def Naive_pitch(pitch_step, pitch_intervals):
    pitch_est = np.zeros((pitch_intervals.shape[0],))

    for i in range(pitch_intervals.shape[0]):
        start_frame = int((pitch_intervals[i][0]-0.01) / 0.02)
        end_frame = int((pitch_intervals[i][1]-0.01) / 0.02)
        if end_frame == start_frame + 1 or end_frame == start_frame:
            pitch_est[i] = pitch_step[start_frame] if pitch_step[start_frame] != 0 else 1.0
        else:
            pitch_est[i] = np.median(pitch_step[start_frame:end_frame]) if np.median(pitch_step[start_frame:end_frame]) != 0 else 1.0

    return pitch_est

def pitch2freq(pitch_np):
    freq_l = [ (2**((pitch_np[i]-69)/12))*440 for i in range(pitch_np.shape[0]) ]
    return np.ndarray(shape=(len(freq_l),), dtype=float, buffer=np.array(freq_l))

def freq2pitch(freq_np):
    pitch_np = 69+12*np.log2(freq_np/440)
    return pitch_np

def minimumEditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if abs(char1 - char2)<0.5:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

def to_semitone(freq):
    return 12*np.log2(freq/440)+69

def eval_note_acc(gt, est):
    gt = to_semitone(gt)
    est = to_semitone(est)
    dist = minimumEditDistance(gt,est)
    note_error_rate = float(dist) / len(est)
    note_accuracy = 1 - note_error_rate
    return note_accuracy

def compute_f1_score(gt_notes, est_notes, delta_p_cents=np.inf, delta_o_sec=np.inf, delta_f_ratio=np.inf):
    """
    Compute F1-score according to VOCANO paper (Section 4.2).
    
    A predicted note is considered correct (true positive) if:
    1. Pitch difference < δp (in cents)
    2. Onset difference < δo (in seconds)
    3. Offset difference < max(δo, δf × g), where g is GT note duration
    
    Args:
        gt_notes: (N, 3) array of [start_time, end_time, frequency] for ground truth
        est_notes: (M, 3) array of [start_time, end_time, frequency] for predictions
        delta_p_cents: pitch tolerance in cents (∞ means not considered)
        delta_o_sec: onset tolerance in seconds (∞ means not considered)
        delta_f_ratio: offset tolerance ratio (∞ means not considered)
    
    Returns:
        dict with 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'
    """
    # Convert frequencies to semitones (MIDI note numbers)
    gt_semitones = to_semitone(gt_notes[:, 2])
    est_semitones = to_semitone(est_notes[:, 2])
    
    matched_gt = set()
    matched_est = set()
    
    # Match each GT note to best matching prediction
    for i, (gt_start, gt_end, gt_freq) in enumerate(gt_notes):
        gt_duration = gt_end - gt_start
        gt_semi = gt_semitones[i]
        
        best_match = None
        best_score = float('inf')
        
        for j, (est_start, est_end, est_freq) in enumerate(est_notes):
            if j in matched_est:
                continue
            
            # Check all three criteria
            pitch_diff_cents = abs(gt_semi - est_semitones[j]) * 100
            onset_diff = abs(est_start - gt_start)
            offset_diff = abs(est_end - gt_end)
            offset_tol = max(delta_o_sec, delta_f_ratio * gt_duration) if delta_f_ratio != np.inf else delta_o_sec
            
            # Check if all criteria are met
            pitch_ok = (delta_p_cents == np.inf) or (pitch_diff_cents <= delta_p_cents)
            onset_ok = (delta_o_sec == np.inf) or (onset_diff <= delta_o_sec)
            offset_ok = (delta_f_ratio == np.inf) or (offset_diff <= offset_tol)
            
            if pitch_ok and onset_ok and offset_ok:
                # Score by total error (for tie-breaking)
                score = onset_diff + offset_diff + pitch_diff_cents / 100.0
                if score < best_score:
                    best_score = score
                    best_match = j
        
        if best_match is not None:
            matched_gt.add(i)
            matched_est.add(best_match)
    
    tp = len(matched_gt)
    fp = len(est_notes) - tp
    fn = len(gt_notes) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'matched_gt_indices': list(matched_gt),
        'matched_est_indices': list(matched_est)
    }

def compare_methods(gt_notes, est_original, est_refined, 
                    onset_tolerance=0.05, offset_tolerance_ratio=0.2, pitch_tolerance_cents=50):
    """
    Comprehensive comparison between original and refined methods using VOCANO paper metrics.
    
    Args:
        gt_notes: (N, 3) array of [start_time, end_time, frequency] for ground truth
        est_original: (M, 3) array of [start_time, end_time, frequency] for original method
        est_refined: (K, 3) array of [start_time, end_time, frequency] for refined method
        onset_tolerance: tolerance for onset matching in seconds (default: 50ms = 0.05)
        offset_tolerance_ratio: ratio of note duration for offset tolerance (default: 0.2)
        pitch_tolerance_cents: tolerance for pitch matching in cents (default: 50 cents = 0.5 semitone)
    
    Returns:
        dict with comparison metrics for both methods following VOCANO paper evaluation
    """
    # Compute F1-scores according to paper (Section 4.2)
    # Main metric: F(50, 0.05, 0.2) - onset-offset-pitch F1-score
    f1_main_orig = compute_f1_score(gt_notes, est_original, 
                                   delta_p_cents=pitch_tolerance_cents,
                                   delta_o_sec=onset_tolerance,
                                   delta_f_ratio=offset_tolerance_ratio)
    
    f1_main_ref = compute_f1_score(gt_notes, est_refined,
                                   delta_p_cents=pitch_tolerance_cents,
                                   delta_o_sec=onset_tolerance,
                                   delta_f_ratio=offset_tolerance_ratio)
    
    # Additional metrics from paper
    f1_onset_only_orig = compute_f1_score(gt_notes, est_original,
                                         delta_p_cents=np.inf,
                                         delta_o_sec=onset_tolerance,
                                         delta_f_ratio=np.inf)
    
    f1_onset_only_ref = compute_f1_score(gt_notes, est_refined,
                                         delta_p_cents=np.inf,
                                         delta_o_sec=onset_tolerance,
                                         delta_f_ratio=np.inf)
    
    f1_offset_only_orig = compute_f1_score(gt_notes, est_original,
                                           delta_p_cents=np.inf,
                                           delta_o_sec=np.inf,
                                           delta_f_ratio=offset_tolerance_ratio)
    
    f1_offset_only_ref = compute_f1_score(gt_notes, est_refined,
                                         delta_p_cents=np.inf,
                                         delta_o_sec=np.inf,
                                         delta_f_ratio=offset_tolerance_ratio)
    
    f1_onset_offset_orig = compute_f1_score(gt_notes, est_original,
                                           delta_p_cents=np.inf,
                                           delta_o_sec=onset_tolerance,
                                           delta_f_ratio=offset_tolerance_ratio)
    
    f1_onset_offset_ref = compute_f1_score(gt_notes, est_refined,
                                          delta_p_cents=np.inf,
                                          delta_o_sec=onset_tolerance,
                                          delta_f_ratio=offset_tolerance_ratio)
    
    f1_onset_pitch_orig = compute_f1_score(gt_notes, est_original,
                                          delta_p_cents=pitch_tolerance_cents,
                                          delta_o_sec=onset_tolerance,
                                          delta_f_ratio=np.inf)
    
    f1_onset_pitch_ref = compute_f1_score(gt_notes, est_refined,
                                         delta_p_cents=pitch_tolerance_cents,
                                         delta_o_sec=onset_tolerance,
                                         delta_f_ratio=np.inf)
    
    # Compute timing and pitch errors for matched notes only
    def compute_errors(gt, est, matched_gt_indices, matched_est_indices):
        """Compute timing and pitch errors for matched notes."""
        onset_errors = []
        offset_errors = []
        pitch_errors = []
        duration_errors = []
        
        gt_semitones = to_semitone(gt[:, 2])
        est_semitones = to_semitone(est[:, 2])
        
        for gt_idx, est_idx in zip(matched_gt_indices, matched_est_indices):
            gt_start, gt_end, gt_freq = gt[gt_idx]
            est_start, est_end, est_freq = est[est_idx]
            
            onset_errors.append(abs(est_start - gt_start))
            offset_errors.append(abs(est_end - gt_end))
            pitch_errors.append(abs(gt_semitones[gt_idx] - est_semitones[est_idx]) * 100)  # cents
            duration_errors.append(abs((est_end - est_start) - (gt_end - gt_start)))
        
        return {
            'mean_onset_error': np.mean(onset_errors) if onset_errors else 0.0,
            'std_onset_error': np.std(onset_errors) if onset_errors else 0.0,
            'mean_offset_error': np.mean(offset_errors) if offset_errors else 0.0,
            'std_offset_error': np.std(offset_errors) if offset_errors else 0.0,
            'mean_pitch_error_cents': np.mean(pitch_errors) if pitch_errors else 0.0,
            'std_pitch_error_cents': np.std(pitch_errors) if pitch_errors else 0.0,
            'mean_duration_error': np.mean(duration_errors) if duration_errors else 0.0,
            'std_duration_error': np.std(duration_errors) if duration_errors else 0.0,
        }
    
    errors_orig = compute_errors(gt_notes, est_original, 
                                 f1_main_orig['matched_gt_indices'],
                                 f1_main_orig['matched_est_indices'])
    
    errors_ref = compute_errors(gt_notes, est_refined,
                                f1_main_ref['matched_gt_indices'],
                                f1_main_ref['matched_est_indices'])
    
    # Sequence-level accuracy
    nacc_original = eval_note_acc(gt_notes[:, 2], est_original[:, 2])
    nacc_refined = eval_note_acc(gt_notes[:, 2], est_refined[:, 2])
    
    # Combine metrics
    metrics_original = {
        **f1_main_orig,
        'f1_onset_only': f1_onset_only_orig['f1'],
        'f1_offset_only': f1_offset_only_orig['f1'],
        'f1_onset_offset': f1_onset_offset_orig['f1'],
        'f1_onset_pitch': f1_onset_pitch_orig['f1'],
        'f1_onset_offset_pitch': f1_main_orig['f1'],  # Main metric
        'nacc': nacc_original,
        'num_notes': len(est_original),
        **errors_orig
    }
    
    metrics_refined = {
        **f1_main_ref,
        'f1_onset_only': f1_onset_only_ref['f1'],
        'f1_offset_only': f1_offset_only_ref['f1'],
        'f1_onset_offset': f1_onset_offset_ref['f1'],
        'f1_onset_pitch': f1_onset_pitch_ref['f1'],
        'f1_onset_offset_pitch': f1_main_ref['f1'],  # Main metric
        'nacc': nacc_refined,
        'num_notes': len(est_refined),
        **errors_ref
    }
    
    # Compute improvements
    improvements = {
        'f1_improvement': metrics_refined['f1_onset_offset_pitch'] - metrics_original['f1_onset_offset_pitch'],
        'f1_onset_only_improvement': metrics_refined['f1_onset_only'] - metrics_original['f1_onset_only'],
        'f1_offset_only_improvement': metrics_refined['f1_offset_only'] - metrics_original['f1_offset_only'],
        'f1_onset_offset_improvement': metrics_refined['f1_onset_offset'] - metrics_original['f1_onset_offset'],
        'f1_onset_pitch_improvement': metrics_refined['f1_onset_pitch'] - metrics_original['f1_onset_pitch'],
        'precision_improvement': metrics_refined['precision'] - metrics_original['precision'],
        'recall_improvement': metrics_refined['recall'] - metrics_original['recall'],
        'onset_error_reduction': metrics_original['mean_onset_error'] - metrics_refined['mean_onset_error'],
        'offset_error_reduction': metrics_original['mean_offset_error'] - metrics_refined['mean_offset_error'],
        'pitch_error_reduction': metrics_original['mean_pitch_error_cents'] - metrics_refined['mean_pitch_error_cents'],
        'nacc_improvement': nacc_refined - nacc_original,
    }
    
    return {
        'original': metrics_original,
        'refined': metrics_refined,
        'improvements': improvements,
        'gt_count': len(gt_notes)
    }

    # INSERT_YOUR_CODE
def load_gt_csv_as_est(gt_csv_path):
    """
    Load a ground truth CSV file in the format:
        start_time,end_time,frequency
        0.974,1.124,293.66
        ...
    and convert it into the same numpy array format as "est":
        ndarray of shape (n,3): [[start, end, freq], ...]
    Args:
        gt_csv_path (str or Path): Path to ground truth csv
    Returns:
        est_like_np (np.ndarray): shape (n, 3), dtype=float
    """
    import csv
    rows = []
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for line in reader:
            if len(line) < 3:
                continue
            start, end, freq = map(float, line[:3])
            rows.append([start, end, freq])
    return np.array(rows, dtype=float)
