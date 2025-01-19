# Models Directory

This directory manages YOLO models for license plate detection.

## Structure

```
models/
└── pre-trained/        # Pre-trained YOLO models
    └── yolo11n.pt     # Base YOLO model for training
```

## Model Management

### Pre-trained Models

The `pre-trained/` directory contains base YOLO models used as starting points for training:
- `yolo11n.pt`: Nano version of YOLOv11, optimized for speed and efficiency
- Additional models can be added as needed

### Training Outputs

When training is run, the following outputs are generated in the run directory:
- `weights/best.pt`: Best model based on validation metrics
- `weights/last.pt`: Model state from the last epoch
- Checkpoints saved every 10 epochs in `weights/`

### Model Selection

The training script will look for models in the following order:
1. Exact path if provided
2. In the `pre-trained/` directory
3. Download from ultralytics if not found locally

### Model Artifacts

Training runs automatically save:
- Model weights
- Training plots
- Performance metrics
- Wandb artifacts for experiment tracking

Once a model is validated you can add it to the `/fine-tuned` directory