import argparse
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.train_vehicle_detection import train_yolo

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for object detection')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data.yaml file')
    
    # Optional arguments with defaults
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                      help='Name or path of the YOLO model to use (default: yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate for training (default: 0.01)')
    parser.add_argument('--project-name', type=str, default='computer-vision-license-plate-detection',
                      help='Project name for wandb logging (default: computer-vision-license-plate-detection)')
    parser.add_argument('--project-dir', type=str, default='src/models/experiments',
                      help='Directory where training runs will be saved (default: src/models/experiments)')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Train the model with provided arguments
    results = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        project_name=args.project_name,
        project_dir=args.project_dir,
        enable_wandb=not args.no_wandb
    )
    
    if results:
        print("\nTraining completed successfully!")
        print(f"Best model saved at: {results['best_model_path']}")
        print("\nFinal Metrics:")
        for metric, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
