"""
YOLO training module for license plate detection.
"""

from ultralytics import YOLO, settings
import torch
import platform
import os
import wandb

# Configure ultralytics to use wandb and optimize performance
settings.update({
    "wandb": True,
})

def get_model_path(model_name):
    """Get the full path to the model file."""
    if os.path.isfile(model_name):
        return model_name
    
    # Check in models/pre-trained directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', 'pre-trained', model_name)
    
    if os.path.isfile(model_path):
        return model_path
    
    # If not found, return the original name (ultralytics will try to download it)
    return model_name

def setup_training_device():
    """Configure the training device and optimize for Apple Silicon."""
    if platform.processor() == 'arm':
        # Enable Metal performance acceleration for M-series chips
        torch.backends.mps.enable = True
        # Set memory format for better performance
        torch.set_float32_matmul_precision('high')
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def on_train_epoch_end(trainer):
    """Log training metrics to wandb after each epoch."""
    metrics = {
        "train/box_loss": float(trainer.loss_items[0]),
        "train/cls_loss": float(trainer.loss_items[1]),
        "train/dfl_loss": float(trainer.loss_items[2]),
        "train/epoch": trainer.epoch,
        "train/lr": float(trainer.optimizer.param_groups[0]['lr'])
    }
    wandb.log(metrics, step=trainer.epoch + 1)

def on_validation_end(validator):
    """Log validation metrics to wandb after each validation."""
    # Get current epoch from trainer (add 1 since epochs are 0-indexed)
    current_epoch = validator.trainer.epoch + 1 if hasattr(validator, 'trainer') else 0
    
    # Get metrics from results dictionary
    results = validator.metrics.results_dict
    metrics = {
        "val/mAP50": float(results['metrics/mAP50(B)']),
        "val/mAP50-95": float(results['metrics/mAP50-95(B)']),
        "val/precision": float(results['metrics/precision(B)']),
        "val/recall": float(results['metrics/recall(B)'])
    }
    wandb.log(metrics, step=current_epoch)

def train_yolo(
    data_yaml,
    model_name="yolo11n.pt",
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    device=None,
    project_name="computer-vision-license-plate-detection",
    project_dir="src/models/experiments",
    enable_wandb=True
):
    """
    Train a YOLO model with the specified parameters.
    
    Args:
        data_yaml (str): Path to the data YAML file
        model_name (str): Name/path of the YOLO model to use
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for training
        device (str, optional): Device to use for training. If None, will be auto-detected
        project_name (str): Name of the project for wandb logging
        project_dir (str): Directory where training runs will be saved
        enable_wandb (bool): Whether to enable wandb logging
    
    Returns:
        dict: Training results and metrics
    """
    if device is None:
        device = setup_training_device()
    
    # Get the full model path
    model_path = get_model_path(model_name)
    print(f"Using model from: {model_path}")
    
    if enable_wandb:
        # Initialize wandb with detailed configuration
        wandb.init(
            project=project_name,
            config={
                "learning_rate": learning_rate,
                "architecture": "YOLOv11",
                "dataset": data_yaml,
                "epochs": epochs,
                "batch_size": batch_size,
                "model_name": model_name,
                "device": str(device),
                "image_size": 640,
                "optimizer": "Adam",
            }
        )
    
    # Initialize model
    model = YOLO(model_path)

    if enable_wandb:
        # Register callbacks only if wandb is enabled
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_end", on_validation_end)

    # Training arguments with optimized settings
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "lr0": learning_rate,
        "device": device,
        "plots": True,
        "save": True,
        "cache": True,
        "imgsz": 640,
        "workers": 14,
        "optimizer": "Adam",
        "save_period": 10,
        "exist_ok": True,
        "patience": 50,
        "project": project_dir,
        "name": project_name
    }

    if enable_wandb:
        train_args["name"] = wandb.run.name

    # Start training
    try:
        results = model.train(**train_args)
        save_dir = model.trainer.save_dir
        
        if enable_wandb:
            # Log final metrics to wandb
            metrics = {
                "final/mAP50": results.maps[0],
                "final/mAP50-95": results.maps[1],
                "final/precision": results.results_dict['metrics/precision(B)'],
                "final/recall": results.results_dict['metrics/recall(B)'],
                "final/box_loss": results.results_dict['train/box_loss'],
                "final/cls_loss": results.results_dict['train/cls_loss'],
                "final/dfl_loss": results.results_dict['train/dfl_loss']
            }
            wandb.log(metrics)
            
            # Log best model as artifact
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                type="model",
                description="Best model from training"
            )
            artifact.add_file(os.path.join(save_dir, 'weights/best.pt'))
            wandb.log_artifact(artifact)
            wandb.finish()
        
        print("\n=== Training Results ===")
        print(f"Training completed. Results saved in: {save_dir}")
        if enable_wandb:
            print(f"WandB run: {wandb.run.name} ({wandb.run.url})")
        print("\nSaved files:")
        print(f"- Best model: {os.path.join(save_dir, 'weights/best.pt')}")
        print(f"- Last model: {os.path.join(save_dir, 'weights/last.pt')}")
        print(f"- Checkpoints: {os.path.join(save_dir, 'weights')}/*.pt")
        
        return {
            "save_dir": save_dir,
            "best_model_path": os.path.join(save_dir, 'weights/best.pt'),
            "last_model_path": os.path.join(save_dir, 'weights/last.pt'),
            "metrics": results.results_dict
        }
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        if enable_wandb:
            wandb.finish()
        return None 