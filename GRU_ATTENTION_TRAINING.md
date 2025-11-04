# GRU Attention Training Guide

## Quick Start

Train the GRU + Attention model using your feature/pitch files:

```bash
python train_gru_attention.py --num_epochs 50 --batch_size 16
```

## Model Architecture

The GRU + Attention model processes audio features through:
1. **1D Convolutions** - Extract frequency patterns
2. **GRU Layers** - Model temporal dependencies  
3. **Multi-Head Attention** - Focus on important time steps
4. **Classification** - Output 6 note segmentation classes

```
Input: (batch, 9, 522, 19) CFP features
    ↓ 1D Convolutions (frequency dimension)
    ↓ GRU (temporal modeling)
    ↓ Multi-Head Attention
    ↓ Classification
Output: (batch, 6) note classes
```

## Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | `gru_attention` | `gru_attention` or `simplified_gru_attention` |
| `--hidden_dim` | 256 | GRU hidden dimension (128, 256, 512, 1024) |
| `--num_layers` | 2 | Number of GRU layers (1, 2, 3, 4) |
| `--learning_rate` | 0.001 | Learning rate (0.0001, 0.001, 0.01) |
| `--batch_size` | 16 | Batch size (8, 16, 32, 64) |
| `--num_epochs` | 50 | Number of training epochs |
| `--feat_dir` | `generated/feat` | Directory with feature .npy files |
| `--pitch_dir` | `generated/pitch` | Directory with pitch .npy files |

## Data Format

The script automatically:
- Loads `.npy` files from `generated/feat/` and `generated/pitch/`
- Reshapes features from `(1566, time_frames)` to `(9, 522, time_frames)`
- Creates sliding windows of size 19
- Converts pitch to binary labels (0=no pitch, 1=pitch)

## Example Commands

### Basic Training
```bash
python train_gru_attention.py --num_epochs 50 --batch_size 16
```

### Custom Parameters
```bash
python train_gru_attention.py \
    --model_type gru_attention \
    --hidden_dim 512 \
    --num_layers 3 \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --num_epochs 100
```

### Fast Testing
```bash
python train_gru_attention.py \
    --model_type simplified_gru_attention \
    --num_epochs 10 \
    --batch_size 32
```

## Model Types

| Model | Parameters | Speed | Best For |
|-------|------------|-------|----------|
| `gru_attention` | ~2M | Medium | Full temporal modeling |
| `simplified_gru_attention` | ~500K | Fast | Quick experiments |

## Parameter Effects

### Hidden Dimension
- **128**: Fast, less capacity (~500K params)
- **256**: Balanced, default (~2M params)  
- **512**: Slow, high capacity (~8M params)
- **1024**: Very slow, very high capacity (~32M params)

### Number of Layers
- **1**: Fast, basic temporal modeling
- **2**: Balanced, good temporal understanding
- **3**: Slow, deep temporal modeling
- **4**: Very slow, very deep modeling

## Training Process

1. **Data Loading**: Automatically loads your feature/pitch files
2. **Preprocessing**: Reshapes and creates sliding windows
3. **Training**: Uses Adam optimizer with learning rate scheduling
4. **Validation**: Monitors accuracy and saves best models
5. **Output**: Saves checkpoints and training curves

## Output Files

- **Model checkpoints**: `checkpoint/epoch_*_gru_attention.pt`
- **Best model**: `checkpoint/best_gru_attention_{model_type}.pt`
- **Final model**: `checkpoint/final_gru_attention_{model_type}.pt`
- **Training curves**: `training_curves.png`

## Using Trained Model

After training, use your model for inference:

```bash
python -m vocano.transcription \
    -n output_name \
    -wd input.wav \
    -mt gru_attention \
    -ckpt checkpoint/best_gru_attention_gru_attention.pt
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 8`
- Use simplified model: `--model_type simplified_gru_attention`

### Slow Training
- Increase batch size: `--batch_size 32`
- Reduce hidden dimension: `--hidden_dim 128`

### Poor Convergence
- Lower learning rate: `--learning_rate 0.0001`
- Increase epochs: `--num_epochs 100`

## Best Practices

1. **Start Simple**: Use default parameters first
2. **Monitor Training**: Watch loss and accuracy curves
3. **Save Checkpoints**: Best models are saved automatically
4. **Resource Management**: Monitor GPU memory usage
5. **Compare Models**: Test different configurations

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm
- Your feature/pitch .npy files in `generated/feat/` and `generated/pitch/`

---

**Ready to train your GRU + Attention model!** 🎵
