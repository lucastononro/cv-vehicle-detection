# License Plate Detection with YOLO

This project implements a machine learning system for license plate detection using YOLO (You Only Look Once) models.

## Folder Structure

```
machine-learning/
├── requirements.txt        # Project dependencies
├── repo-structure.md      # Detailed repository documentation
└── src/                   # Source code
    ├── data/             # Dataset directory
    ├── data-prep/        # Data preparation scripts
    ├── models/           # Model management
    │   ├── pre-trained/  # Pre-trained base models
    │   ├── fine-tuned/   # Best performing trained models - manually moved to production
    │   └── experiments/  # Training experiments and results
    └── train/           # Training implementation
```

See `repo-structure.md` for detailed documentation description of folders.

## Setup Weights & Biases (WandB)

1. Install WandB:
```bash
pip install wandb
```

2. Login to WandB:
```bash
wandb login
```
You'll be prompted to enter your API key. You can find this at https://wandb.ai/settings

3. (Optional) Configure WandB defaults:
```bash
wandb init
```
This will help you set up your default project name and other settings.

4. Verify installation:
```bash
python -c "import wandb; print(wandb.__version__)"
```

Notes:
- First time you run training, WandB will open a browser window for authentication
- You can find your experiments at https://wandb.ai/your-username/your-project
- Use `--no-wandb` flag if you want to train without logging
- Each run creates a unique experiment in your WandB project
- Metrics are logged in real-time during training

## How to Train the Model

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
- Place images in `src/data/vehicle-detection-639on/train/images/`
- Place labels in `src/data/vehicle-detection-639on/train/labels/`
- Ensure `data.yaml` is configured correctly

3. Start training:
```bash
python src/train.py \
    --data src/data/vehicle-detection-639on/data.yaml \
    --model yolo11n.pt \
    --epochs 300 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --project-dir src/models/experiments \
    --project-name "computer-vision-license-plate"
```

4. Monitor training:
- Open Weights & Biases dashboard to track progress
  - Go to https://wandb.ai/your-username/computer-vision-license-plate-detection
  - View real-time plots of loss and metrics
  - Compare different training runs
  - Download saved models and artifacts
- Check console output for real-time metrics
- Models are saved in the project directory under `src/models/experiments/[run-name]/weights/`

## Training Configuration

Key parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.01)
- `--model`: Model to use (default: yolo11n.pt)
- `--project-name`: WandB project name
- `--project-dir`: Directory for saving runs (default: src/models/experiments)
- `--no-wandb`: Disable WandB logging

## Model Management

Training outputs are organized as follows:
```
src/models/
├── pre-trained/          # Base models for training
│   └── yolo11n.pt       # Default YOLO model
├── fine-tuned/          # Production-ready models
│   └── [version]/       # Each validated model
│       ├── model.pt     # Model weights
│       └── metrics.json # Performance metrics
└── experiments/         # Training experiments
    └── [run-name]/     # Each training run
        └── weights/    # Model weights and checkpoints
            ├── best.pt    # Best model by validation metrics
            ├── last.pt    # Final model state
            └── epoch_*.pt # Regular checkpoints
```

### Model Workflow

1. Start with a pre-trained model from `pre-trained/`
2. Train model with custom data, outputs go to `experiments/`
3. Evaluate and validate the trained model
4. If performance is good, move to `fine-tuned/` with version and metrics
5. Use models from `fine-tuned/` for production inference

## Model Outputs

Training produces:
- Best model: `weights/best.pt`
- Final model: `weights/last.pt`
- Checkpoints: Saved every 10 epochs
- Training plots and metrics
- WandB artifacts and logs

## Hardware Support

Automatically optimizes for:
- Apple Silicon (M1/M2) with MPS
- NVIDIA GPUs with CUDA
- CPU fallback when needed

## Testing (Planned)

The testing module will provide:
- Model inference
- Performance evaluation
- Visualization tools
- Batch processing

## Dependencies

Core requirements:
- ultralytics: YOLO implementation
- torch: Deep learning framework
- wandb: Experiment tracking
- opencv-python: Image processing
- numpy: Numerical operations

See `requirements.txt` for complete list.

# Machine Learning

This directory contains the machine learning components for training and evaluating vehicle detection models.

## Data Setup

### 1. Download Dataset

You can obtain the training data from:
- [Roboflow Universe](https://universe.roboflow.com/) (recommended)
- Any other dataset compatible with YOLO format

For Roboflow:
1. Create an account at [Roboflow](https://roboflow.com)
2. Find or create a vehicle detection dataset
3. Export the dataset in YOLOv8 format
4. Download and extract to `src/data/your-dataset-name/`

### 2. Configure data.yaml

After downloading the dataset, update the `data.yaml` file in your dataset directory:

```yaml
# src/data/your-dataset-name/data.yaml

train: path/to/your/train/images  # Path to training images
val: path/to/your/valid/images    # Path to validation images
test: path/to/your/test/images    # Path to test images (optional)

nc: 1  # Number of classes
names: ['vehicle']  # Class names
```

Important notes:
- Use absolute paths or paths relative to where you'll run training
- Verify that the paths point to the correct image directories
- Update `nc` and `names` according to your classes
- The default structure expects:
  ```
  src/data/your-dataset-name/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml
  ```

