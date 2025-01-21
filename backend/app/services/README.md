# Services

This directory contains the core business logic and services for the Vehicle Detection system.

## Structure

```
services/
├── models/              # ML model implementations
│   ├── base_model.py   # Base class for all models
│   ├── yolo_model.py   # YOLO model implementation
│   └── model_factory.py # Factory for model instantiation
├── inference.py        # Inference service for detection
└── __init__.py        # Service initialization
```

## Components

### Model Services

- `base_model.py`: Abstract base class defining the interface for all detection models
- `yolo_model.py`: YOLO model implementation for vehicle detection
- `model_factory.py`: Factory pattern implementation for model instantiation

### Inference Service

The `inference.py` module provides:
- Video frame processing
- Vehicle detection pipeline
- Result aggregation and processing
- Batch processing capabilities

## Usage

```python
from app.services.models.model_factory import ModelFactory
from app.services.inference import InferenceService

# Initialize model
model = ModelFactory.create_model("yolo")

# Create inference service
inference_service = InferenceService(model)

# Process video frame
results = await inference_service.process_frame(frame)
```

## Adding New Models

To add a new model:

1. Create a new model class in `models/` that inherits from `BaseModel`
2. Implement required methods:
   - `load_model()`
   - `predict()`
   - `preprocess()`
   - `postprocess()`
3. Register the model in `model_factory.py` 