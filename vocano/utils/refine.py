"""
Sub-frame onset/offset refinement using spectral flux from CFP feature.

Context
- The baseline decoding (Smooth_sdt6_modified) produces note intervals on a 20 ms grid,
  encoded at frame centers: t = 0.02 * frame + 0.01.
- This utility refines those boundary times (onsets/offsets) between frames without
  changing the detected note count or ordering.

Approach
- Build a per-frame spectral flux curve from the CFP feature (we use the first 522 rows
  that correspond to 3-scale spectral-flux-like channels SN1/SN2/SN3).
- Around each boundary, take a small window (±window_frames), upsample the flux by
  linear interpolation, and pick a sub-frame time via:
  - peak: sub-frame argmax in the local window
  - cross: first rising threshold crossing at alpha * local_max
- Constrain the refined time to lie between the centers of the two adjacent 20 ms frames.

Usage
    refined = refine_intervals_with_flux(feature, pitch_intervals)
    # feature: (1566, T) from full_feature_extraction
    # pitch_intervals: (N, 2) from Smooth_sdt6_modified

Notes
- This is a post-process; no model retraining needed.
- For best precision, you can increase upsample or window_frames modestly.
"""

import numpy as np
from typing import Tuple

HOP_SECONDS = 0.02
CENTER_OFFSET = 0.01


def _frame_time(frame_idx: int) -> float:
	return HOP_SECONDS * frame_idx + CENTER_OFFSET


def _time_to_frame(time_s: float) -> int:
	# Inverse of encoding in Smooth_sdt6_modified
	return int((time_s - CENTER_OFFSET) / HOP_SECONDS)


def _extract_flux_from_feature(feature: np.ndarray) -> np.ndarray:
	"""
	Build a simple per-frame spectral flux proxy from the CFP feature.

	feature layout: (1566, T) stacked as [SN(522), SIN(522), ZN(522)]
	- SN contains 3 x 174 spectral-flux-like bands across three time scales
	- We sum SN over frequency to get a robust scalar flux per frame

	Returns:
		flux: shape (T,), larger around onsets
	"""
	assert feature.ndim == 2 and feature.shape[0] >= 522, "feature must be (1566, T) or at least start with SN (522, T)"
	SN = feature[:522, :]  # 3 scales x 174 bins
	flux = SN.sum(axis=0)
	# Normalize to unit variance locally later; keep raw here
	return flux


def _refine_boundary_time(
	flux: np.ndarray,
	boundary_frame: int,
	window_frames: int = 2,
	upsample: int = 10,
	method: str = "peak",
	alpha: float = 0.3,
) -> float:
	"""
	Refine a boundary near boundary_frame using local spectral flux.

	Parameters
	- boundary_frame: integer frame index from coarse decoding
	- window_frames: how many frames to include on each side for local search
	- upsample: linear interpolation factor for sub-frame resolution
	- method:
	  - "peak": choose sub-frame argmax of flux
	  - "cross": choose first rising crossing at alpha * local_max
	- alpha: threshold factor for the crossing method

	The result is constrained within [center(boundary-1), center(boundary+1)]
	to avoid changing note count by large shifts.
	"""
	T = flux.shape[0]
	f0 = boundary_frame
	lo = max(0, f0 - window_frames)
	hi = min(T - 1, f0 + window_frames)

	# Build sampled points and normalize locally
	t = np.arange(lo, hi + 1, dtype=float)
	y = flux[lo : hi + 1].astype(float)
	if y.std() > 1e-8:
		y = (y - y.mean()) / y.std()

	# Upsample via simple linear interpolation (dependency-free)
	if upsample > 1 and len(t) >= 2:
		fine_t = np.linspace(t[0], t[-1], num=(len(t) - 1) * upsample + 1)
		fine_y = np.interp(fine_t, t, y)
	else:
		fine_t = t
		fine_y = y

	# Choose sub-frame position
	if method == "peak":
		idx = int(np.argmax(fine_y))
		best_frame = fine_t[idx]
	elif method == "cross":
		local_max = float(fine_y.max()) if fine_y.size else 0.0
		th = alpha * local_max
		best_frame = float(f0)
		for i in range(1, fine_y.size):
			if fine_y[i - 1] < th <= fine_y[i]:
				# linear interpolation between i-1 and i
				w = (th - fine_y[i - 1]) / (fine_y[i] - fine_y[i - 1] + 1e-12)
				best_frame = fine_t[i - 1] * (1 - w) + fine_t[i] * w
				break
	else:
		best_frame = float(f0)

	# Constrain within adjacent centers
	min_time = _frame_time(max(f0 - 1, 0))
	max_time = _frame_time(min(f0 + 1, T - 1))
	best_time = _frame_time(best_frame)
	best_time = min(max(best_time, min_time), max_time)
	return best_time


def refine_intervals_with_flux(
	feature: np.ndarray,
	pitch_intervals: np.ndarray,
	window_frames: int = 2,
	upsample: int = 10,
	method: str = "peak",
	alpha: float = 0.3,
) -> np.ndarray:
	"""
	Refine onset/offset timestamps in pitch_intervals using spectral flux from CFP feature.

	Args:
		feature: (1566, T) CFP feature from full_feature_extraction
		pitch_intervals: (N, 2) array of [start_time, end_time] (seconds)
		window_frames: +/- frames around boundary to consider
		upsample: interpolation factor for sub-frame search
		method: 'peak' or 'cross' (threshold crossing)
		alpha: threshold factor for 'cross' in [0,1]

	Returns:
		(N, 2) array of refined [start_time, end_time] in seconds
	"""
	if pitch_intervals is None or len(pitch_intervals) == 0:
		return pitch_intervals

	flux = _extract_flux_from_feature(feature)
	refined = pitch_intervals.copy().astype(float)

	for i in range(refined.shape[0]):
		# Start
		start_frame = _time_to_frame(refined[i, 0])
		refined[i, 0] = _refine_boundary_time(
			flux, start_frame, window_frames=window_frames, upsample=upsample, method=method, alpha=alpha
		)
		# End
		end_frame = _time_to_frame(refined[i, 1])
		refined[i, 1] = _refine_boundary_time(
			flux, end_frame, window_frames=window_frames, upsample=upsample, method=method, alpha=alpha
		)
		# Ensure valid ordering
		if refined[i, 1] <= refined[i, 0]:
			# If invalid due to refinement, keep original
			refined[i, :] = pitch_intervals[i, :]

	return refined


