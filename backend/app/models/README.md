# Models

This directory contains the trained ML models used for vehicle detection.

## Model Files

Add the model file here (.pt from yolovX) and the model will be handled by the model factory

## Usage

The models are automatically loaded by the `ModelFactory` in the services layer:

```python
from app.services.models.model_factory import ModelFactory

# Load vehicle detection model
vehicle_model = ModelFactory.create_model("yolo", model_path="models/epoch50.pt")

# Load license plate model
plate_model = ModelFactory.create_model("yolo", model_path="models/license_plate_detector.pt")

# Load lightweight model for mobile/edge devices
mobile_model = ModelFactory.create_model("yolo", model_path="models/yolo11n.pt")
```

## Updating Models

To update a model:

1. Place the new model file in this directory
2. Restart the application