## AINoteTranscription (VOCANO-based)

This project is inherited from the original VOCANO: A note transcription framework for singing voice in polyphonic music. We keep the core pipeline (Patch-CNN pitch extraction + note segmentation), modernize the runtime, and add experimental options.

### What’s different from upstream VOCANO
- **Modern dependencies**: requirements updated for current CUDA/PyTorch and libraries (see requirements below). No NVIDIA Apex is required; uses PyTorch AMP when enabled.
- **Checkpoints auto-download**: model files are fetched via Google Drive on first run.
- **Experimental models**: optional GRU + Attention note segmentation variants are wired in but require training (see GRU_ATTENTION_TRAINING.md).
- **Utilities for accuracy**: added `vocano/utils/refine.py` to refine onsets/offsets for better Note Accuracy (NAcc).

## Requirements (modernized)

Tested on Python 3.10+ with CUDA 12.x. Key pinned packages (see `requirements.txt`):
- torch==2.8.0+cu124, torchvision==0.23.0+cu124 (from the CUDA 12.4 extra index)
- numpy==2.3.3, scipy==1.16.2
- librosa==0.11.0, pretty-midi==0.2.9
- cupy-cuda12x==13.6.0 (optional, enables GPU CFP feature extraction)
- tqdm, matplotlib, pandas, googledrivedownloader

Install with:
```bash
pip install -r requirements.txt
```

Notes:
- AMP/mixed precision uses native PyTorch; Apex is not needed.
- If you install a different CUDA build, adjust the PyTorch wheels accordingly.

## Quick start

1) Put your input audio `.wav` somewhere on disk.

2) Run the transcription entrypoint (end-to-end):
```bash
python -m vocano.transcription -n output_name -wd path/to/input.wav
```

Useful flags (as parsed by `vocano/core.py`):
- `-d cpu|<gpu_id>|auto` device selection (`auto` supported on Linux only).
- `-use_cp` enable CuPy-accelerated CFP extraction when `cupy-cuda12x` is installed.
- `-s` save extracted features/pitch to `generated/feat` and `generated/pitch`.
- `-use_pre` reuse previously extracted features/pitch for faster runs.
- `-ckpt` path to a model checkpoint (defaults are auto-downloaded when using PyramidNet).
- `-mt`/`-model_type` choose `pyramidnet` (default), `gru_attention`, or `simplified_gru_attention`.
- `-bz` `-nw` `-pn` batch size, workers, and pin-memory for dataloaders.

Outputs are written to:
- MIDI: `generated/midi/<name>.mid`
- Synthesized audio: `generated/wav/<name>.wav`
- Raw model logits sample: `ond/<name>_raw_outputs.npy|.csv` (for inspection)

## Pipeline overview

The VOCANO inference decomposes SVT into two parts:
- Pitch extraction: Patch-CNN (checkpoint auto-downloaded).
- Note segmentation: PyramidNet-110 with ShakeDrop (default), or experimental GRU+Attention variants.

The `vocano/core.py` class orchestrates:
- model/file downloads, device selection, CFP feature extraction (NumPy or CuPy), inference with optional AMP, post-processing (Smooth_sdt6_modified), and MIDI synthesis.

## Improving Note Accuracy (NAcc) with sub-frame refinement

File: `vocano/utils/refine.py`

Purpose: refine onset/offset boundary times between 20 ms frames using a spectral-flux proxy from the CFP feature, yielding tighter boundaries and potentially higher NAcc.

How it works (summary):
- Build a per-frame spectral-flux curve from the first 522 CFP rows.
- Around each coarse boundary, take a small window, upsample by linear interpolation, and pick a sub-frame time by either peak or threshold-crossing.
- Constrain results between adjacent frame centers to preserve note count and ordering.

Example usage after you obtain `feature` and `pitch_intervals` from the normal pipeline:
```python
from vocano.utils.refine import refine_intervals_with_flux

refined = refine_intervals_with_flux(
    feature=feature,               # (1566, T) CFP
    pitch_intervals=pitch_intervals,  # (N, 2) [start_s, end_s]
    window_frames=2,
    upsample=10,
    method="peak",               # or "cross"
    alpha=0.3                     # for crossing threshold
)
# Use `refined` in place of `pitch_intervals` for MIDI/export/evaluation.
```

This utility is not mandatory for running the baseline, but is provided as an accuracy-oriented post-process step.

## Experimental: GRU + Attention note segmentation (training required)

We expose optional GRU+Attention models in `vocano/core.py` as `-mt gru_attention` or `-mt simplified_gru_attention`. These models need training before meaningful inference. See:
- `GRU_ATTENTION_TRAINING.md` — training guide, options, outputs, and usage.

Until you train and provide `-ckpt` for these models, the code will fall back to random initialization and warn you.

## Tips

- For polyphonic audio, consider separating vocals first (e.g., Demucs) and run transcription on the vocal stem for better results.
- Windows users: `-d auto` is Linux-only; specify `-d cpu` or a concrete GPU id on Windows.
- CuPy acceleration is optional; if memory is insufficient, the code automatically falls back to NumPy and frees GPU memory between steps.

## Citation

Please cite the original VOCANO and Omnizart works:

```
@inproceedings{vocano,
  title={{VOCANO}: A Note Transcription Framework For Singing Voice In Polyphonic Music},
  author={Hsu, Jui-Yang and Su, Li},
  booktitle={Proc. International Society of Music Information Retrieval Conference (ISMIR)},
  year={2021}
}
```

```
@article{wu2021omnizart,
  title={Omnizart: A General Toolbox for Automatic Music Transcription},
  author={Wu, Yu-Te and Luo, Yin-Jyun and Chen, Tsung-Ping and Wei, I-Chieh and Hsu, Jui-Yang and Chuang, Yi-Chin and Su, Li},
  journal={arXiv preprint arXiv:2106.00497},
  year={2021}
}
```