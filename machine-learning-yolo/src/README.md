# Source Directory

This directory contains the core implementation of the license plate detection system.

## Directory Structure

```
src/
├── data/                # Dataset directory
├── data-prep/          # Data preparation scripts
├── models/             # Model storage
├── train/             # Training implementation
└── test/              # Testing implementation
```

## Components

### Data Preparation (`data-prep/`)

Contains scripts for preparing and converting dataset labels:
- `convert_labels-vehicle-detection-639on.py`: Converts and normalizes label formats
  - Merges "license" and "licenseplate" labels into a single class
  - Processes train/valid/test splits
  - Ensures consistent label format across the dataset

### Training (`train/`)

Training implementation for YOLO models:
- Main training script with CLI interface
- Optimized for different hardware (CPU, GPU, Apple Silicon)
- Experiment tracking with Weights & Biases
- Automatic model checkpointing

### Models (`models/`)

Storage for:
- Pre-trained models
- Training checkpoints
- Best and last model weights
- Model artifacts

### Testing (`test/`)

Contains:
- Inference scripts
- Model evaluation tools
- Performance metrics calculation
- Visualization utilities


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