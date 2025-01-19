# Machine Learning Repository Structure

This directory contains the machine learning components for license plate detection using YOLO (You Only Look Once) models.

## Directory Structure

```
machine-learning/
├── requirements.txt        # Project dependencies
└── src/
    ├── data/             # Dataset directory
    │   └── vehicle-detection-639on/
    │       ├── train/    # Training images and labels
    │       ├── valid/    # Validation images and labels
    │       ├── test/     # Test images and labels
    │       └── data.yaml # Dataset configuration
    ├── data-prep/        # Data preparation scripts
    ├── models/           # Model management
    │   ├── pre-trained/  # Pre-trained base models
    │   ├── fine-tuned/   # Best performing trained models
    │   └── experiments/  # Training experiments and results
    └── train/           # Training scripts
        ├── __init__.py
        └── train_vehicle_detection.py
```

## Components

### Training (`src/train/`)

The training module provides a robust implementation for training YOLO models on license plate detection:

- `train.py`: Main entry point for training
  - Provides CLI interface for training configuration
  - Supports customizable hyperparameters
  - Integrates with Weights & Biases for experiment tracking

- `train/train_vehicle_detection.py`: Core training implementation
  - Implements YOLO model training with optimizations
  - Supports Apple Silicon (M1/M2) acceleration
  - Includes callbacks for metric logging
  - Handles model checkpointing and artifacts

### Testing (`src/test/`) [To Be Implemented]

The testing module will provide:
- Model inference on new images/videos
- Performance evaluation metrics
- Visualization tools for detection results
- Batch processing capabilities

### Data Structure (`src/data/`)

The dataset follows the YOLO format:
- Images: JPG/PNG format
- Labels: Text files with normalized bounding box coordinates
- `data.yaml`: Configuration file specifying:
  - Training/validation/test splits
  - Class names
  - Number of classes
  - Path configurations

### Model Management (`src/models/`)

The models directory organizes all model-related files:

- `pre-trained/`: Base models for training
  - Contains initial YOLO models (e.g., yolo11n.pt)
  - Starting point for transfer learning

- `fine-tuned/`: Production-ready models
  - Contains best performing trained models
  - Validated and ready for deployment
  - Versioned with performance metrics
  - Used for inference and testing

- `experiments/`: Training outputs
  - Each run gets a unique directory
  - Contains weights, checkpoints, and metrics
  - Organized by experiment name and date
  - Used to track training progress
  - Best models should be moved to `fine-tuned/` after validation

## Usage

### Training

```bash
python src/train.py \
    --data src/data/vehicle-detection-639on/data.yaml \
    --model yolo11n.pt \
    --epochs 300 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --project-dir models/experiments \
    --project-name "computer-vision-license-plate-detection"
```

### Experiment Tracking

Training progress is tracked using Weights & Biases:
- Loss metrics (box, classification, DFL)
- Validation metrics (mAP50, mAP50-95, precision, recall)
- Learning rate scheduling
- Model checkpoints as artifacts

## Dependencies

Key dependencies (see requirements.txt):
- ultralytics: YOLO implementation
- torch: Deep learning framework
- wandb: Experiment tracking
- opencv-python: Image processing
- numpy: Numerical operations 