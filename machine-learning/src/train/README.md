# Training Implementation

This directory contains the YOLO training implementation for license plate detection.

## Files

```
train/
├── __init__.py                    # Package initialization
└── train_vehicle_detection.py     # Core training implementation
```

## Training Features

### Hardware Optimization

- Automatic device detection and configuration:
  - Apple Silicon (M1/M2) with MPS acceleration
  - NVIDIA GPUs with CUDA
  - Fallback to CPU when needed

### Training Configuration

- Flexible hyperparameter tuning:
  - Learning rate
  - Batch size
  - Number of epochs
  - Model architecture
  - Image size
  - Optimizer settings

### Experiment Tracking

Comprehensive metric logging with Weights & Biases:

Training Metrics:
- Box loss
- Classification loss
- DFL loss
- Learning rate

Validation Metrics:
- mAP50
- mAP50-95
- Precision
- Recall

### Model Checkpointing

- Saves best model based on validation metrics
- Keeps last model state
- Regular checkpoints every 10 epochs
- Early stopping with patience=50

## Usage

Basic training command:
```bash
python src/train.py \
    --data src/data/vehicle-detection-639on/data.yaml \
    --model yolo11n.pt \
    --epochs 300 \
    --batch-size 64 \
    --learning-rate 0.001
```

Additional options:
- `--project-name`: Custom name for wandb tracking
- `--no-wandb`: Disable wandb logging

## Model Selection

The training script automatically handles model loading:
1. Checks for local file path
2. Looks in pre-trained models directory
3. Downloads from ultralytics if needed
