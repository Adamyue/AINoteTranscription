"""
Sub-frame onset/offset refinement for 20ms frame-based note transcription.

Context
- The baseline decoding (Smooth_sdt6_modified) produces note intervals on a 20 ms grid,
  encoded at frame centers: t = 0.02 * frame + 0.01.
- This utility refines those boundary times (onsets/offsets) between frames without
  changing the detected note count or ordering.

Available Refinement Methods

1. Model Probability Curves (RECOMMENDED)
   - Uses onSeq/offSeq from Smooth_sdt6_modified (already computed model outputs)
   - Most direct: the model already learned onset/offset patterns
   - Methods: peak, parabolic, weighted_centroid, threshold_crossing
   - Best for pure vocal audio

2. Combined Method (Model Probabilities + Pitch)
   - Weighted combination of model probabilities and pitch contour derivative
   - Pitch derivative helps with vocal note boundaries
   - More robust but requires tuning weights

Usage
    # Method 1: Using model probabilities (recommended for vocals)
    refined = refine_intervals_with_probabilities(
        pitch_intervals, onSeq_np, offSeq_np, method='parabolic')
    
    # Method 2: Combined with pitch
    refined = refine_intervals_combined(
        pitch_intervals, onSeq_np, offSeq_np, pitch=pitch)

Notes
- All methods constrain results within [center(f-1), center(f+1)] to preserve note count
- Parabolic fitting is generally more accurate than simple peak picking
- Model probabilities are the most reliable for vocal transcription
"""

import numpy as np

HOP_SECONDS = 0.02
CENTER_OFFSET = 0.01


def _frame_time(frame_idx: int) -> float:
	return HOP_SECONDS * frame_idx + CENTER_OFFSET


def _time_to_frame(time_s: float) -> int:
	# Inverse of encoding in Smooth_sdt6_modified
	return int((time_s - CENTER_OFFSET) / HOP_SECONDS)


def _parabolic_peak_fit(t: np.ndarray, y: np.ndarray) -> float:
	"""
	Fit a parabola to 3 points around the maximum and find its vertex.
	More accurate than simple argmax for sub-frame positioning.
	
	Returns the frame index of the parabola vertex.
	"""
	if len(t) < 3:
		return float(t[np.argmax(y)])
	
	# Find peak index
	peak_idx = int(np.argmax(y))
	# Use peak and neighbors (clamp to valid range)
	i0 = max(0, peak_idx - 1)
	i1 = peak_idx
	i2 = min(len(t) - 1, peak_idx + 1)
	
	# Extract 3 points
	t0, t1, t2 = t[i0], t[i1], t[i2]
	y0, y1, y2 = y[i0], y[i1], y[i2]
	
	# Fit parabola: y = a*t^2 + b*t + c
	# Solve using Lagrange interpolation or direct formula
	denom = (t0 - t1) * (t0 - t2) * (t1 - t2)
	if abs(denom) < 1e-10:
		return float(t1)
	
	# Compute coefficients
	a = ((t2 - t1) * y0 + (t0 - t2) * y1 + (t1 - t0) * y2) / denom
	b = ((t1**2 - t2**2) * y0 + (t2**2 - t0**2) * y1 + (t0**2 - t1**2) * y2) / denom
	
	# Vertex: t_vertex = -b / (2*a)
	if abs(a) > 1e-10:
		t_vertex = -b / (2.0 * a)
		# Constrain to valid range
		t_vertex = max(t[i0], min(t_vertex, t[i2]))
		return float(t_vertex)
	else:
		return float(t1)


def _weighted_centroid(t: np.ndarray, y: np.ndarray) -> float:
	"""
	Compute weighted centroid of the probability distribution.
	Works well for broad peaks.
	"""
	if y.sum() < 1e-10:
		return float(t[len(t) // 2])
	# Normalize to probabilities
	y_norm = y / (y.sum() + 1e-10)
	centroid = (t * y_norm).sum()
	return float(centroid)


def _refine_boundary_with_probability(
	prob_seq: np.ndarray,
	boundary_frame: int,
	window_frames: int = 2,
	upsample: int = 10,
	method: str = "parabolic",
	alpha: float = 0.3,
) -> float:
	"""
	Refine a boundary using model probability curves (onSeq/offSeq).
	
	Parameters
	- prob_seq: per-frame probability sequence (already smoothed)
	- boundary_frame: integer frame index from coarse decoding
	- window_frames: how many frames to include on each side
	- upsample: interpolation factor for sub-frame resolution
	- method:
	  - "parabolic": fit parabola to peak (RECOMMENDED)
	  - "weighted_centroid": weighted centroid of probability mass
	  - "peak": simple argmax
	  - "cross": threshold crossing
	- alpha: threshold factor for crossing method
	"""
	T = prob_seq.shape[0]
	f0 = boundary_frame
	lo = max(0, f0 - window_frames)
	hi = min(T - 1, f0 + window_frames)
	
	# Extract local window
	t = np.arange(lo, hi + 1, dtype=float)
	y = prob_seq[lo : hi + 1].astype(float)
	
	# Upsample via linear interpolation
	if upsample > 1 and len(t) >= 2:
		fine_t = np.linspace(t[0], t[-1], num=(len(t) - 1) * upsample + 1)
		fine_y = np.interp(fine_t, t, y)
	else:
		fine_t = t
		fine_y = y
	
	# Choose refinement method
	if method == "parabolic":
		best_frame = _parabolic_peak_fit(fine_t, fine_y)
	elif method == "weighted_centroid":
		best_frame = _weighted_centroid(fine_t, fine_y)
	elif method == "peak":
		idx = int(np.argmax(fine_y))
		best_frame = fine_t[idx]
	elif method == "cross":
		local_max = float(fine_y.max()) if fine_y.size else 0.0
		th = alpha * local_max
		best_frame = float(f0)
		for i in range(1, fine_y.size):
			if fine_y[i - 1] < th <= fine_y[i]:
				w = (th - fine_y[i - 1]) / (fine_y[i] - fine_y[i - 1] + 1e-12)
				best_frame = fine_t[i - 1] * (1 - w) + fine_t[i] * w
				break
	else:
		best_frame = float(f0)
	
	# Constrain within adjacent frame centers
	min_time = _frame_time(max(f0 - 1, 0))
	max_time = _frame_time(min(f0 + 1, T - 1))
	best_time = _frame_time(best_frame)
	best_time = min(max(best_time, min_time), max_time)
	return best_time


def refine_intervals_with_probabilities(
	pitch_intervals: np.ndarray,
	onSeq_np: np.ndarray,
	offSeq_np: np.ndarray,
	window_frames: int = 2,
	upsample: int = 10,
	method: str = "parabolic",
	alpha: float = 0.3,
) -> np.ndarray:
	"""
	Refine onset/offset timestamps using model probability curves.
	
	This is the RECOMMENDED method as it uses the model's own
	onset/offset predictions directly.
	
	Args:
		pitch_intervals: (N, 2) array of [start_time, end_time] (seconds)
		onSeq_np: (T,) onset probability sequence from Smooth_sdt6_modified
		offSeq_np: (T,) offset probability sequence from Smooth_sdt6_modified
		window_frames: +/- frames around boundary to consider
		upsample: interpolation factor for sub-frame search
		method: 'parabolic' (recommended), 'weighted_centroid', 'peak', 'cross'
		alpha: threshold factor for 'cross' method
	
	Returns:
		(N, 2) array of refined [start_time, end_time] in seconds
	"""
	if pitch_intervals is None or len(pitch_intervals) == 0:
		return pitch_intervals
	
	refined = pitch_intervals.copy().astype(float)
	
	for i in range(refined.shape[0]):
		# Refine onset (start_time) using onSeq
		start_frame = _time_to_frame(refined[i, 0])
		refined[i, 0] = _refine_boundary_with_probability(
			onSeq_np, start_frame,
			window_frames=window_frames, upsample=upsample,
			method=method, alpha=alpha
		)
		
		# Refine offset (end_time) using offSeq
		end_frame = _time_to_frame(refined[i, 1])
		refined[i, 1] = _refine_boundary_with_probability(
			offSeq_np, end_frame,
			window_frames=window_frames, upsample=upsample,
			method=method, alpha=alpha
		)
		
		# Ensure valid ordering
		if refined[i, 1] <= refined[i, 0]:
			refined[i, :] = pitch_intervals[i, :]
	
	return refined


def refine_intervals_combined(
	pitch_intervals: np.ndarray,
	onSeq_np: np.ndarray,
	offSeq_np: np.ndarray,
	pitch: np.ndarray = None,
	weights: dict = None,
	window_frames: int = 2,
	upsample: int = 10,
) -> np.ndarray:
	"""
	Refine intervals using weighted combination of model probabilities and pitch derivative.
	
	Args:
		pitch_intervals: (N, 2) array of [start_time, end_time]
		onSeq_np, offSeq_np: model probability sequences
		pitch: (T,) pitch contour (optional, for pitch derivative)
		weights: dict with keys 'prob', 'pitch' (default: {'prob': 0.7, 'pitch': 0.3})
		window_frames, upsample: refinement parameters
	
	Returns:
		(N, 2) refined intervals
	"""
	if weights is None:
		weights = {'prob': 0.7, 'pitch': 0.3}
	
	# Normalize weights
	total = sum(v for k, v in weights.items() if v > 0)
	weights = {k: v / total for k, v in weights.items()}
	
	refined = pitch_intervals.copy().astype(float)
	
	# Build combined signals
	combined_on = onSeq_np.copy()
	combined_off = offSeq_np.copy()
	
	if pitch is not None and weights.get('pitch', 0) > 0:
		# Compute pitch derivative (absolute change)
		pitch_diff = np.abs(np.diff(pitch, prepend=pitch[0]))
		pitch_diff_norm = (pitch_diff - pitch_diff.min()) / (pitch_diff.max() - pitch_diff.min() + 1e-10)
		combined_on = weights['prob'] * onSeq_np + weights['pitch'] * pitch_diff_norm
		combined_off = weights['prob'] * offSeq_np + weights['pitch'] * pitch_diff_norm
	
	# Refine using combined signals
	for i in range(refined.shape[0]):
		start_frame = _time_to_frame(refined[i, 0])
		refined[i, 0] = _refine_boundary_with_probability(
			combined_on, start_frame, window_frames=window_frames,
			upsample=upsample, method='parabolic'
		)
		
		end_frame = _time_to_frame(refined[i, 1])
		refined[i, 1] = _refine_boundary_with_probability(
			combined_off, end_frame, window_frames=window_frames,
			upsample=upsample, method='parabolic'
		)
		
		if refined[i, 1] <= refined[i, 0]:
			refined[i, :] = pitch_intervals[i, :]
	
	return refined

