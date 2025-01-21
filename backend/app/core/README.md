# Core

This directory contains core configuration and settings for the Vehicle Detection system.

## Files

- `config.py`: Application configuration using Pydantic settings management
  - Environment variables configuration
  - AWS S3 settings
  - Model paths and configurations
  - API settings

## Usage

```python
from app.core.config import settings

# Access configuration
S3_ENDPOINT = settings.S3_ENDPOINT_URL
MODEL_PATH = settings.MODEL_PATH
```

## Configuration Variables

The following environment variables can be configured:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_ENDPOINT_URL=your_s3_endpoint

# Model Configuration
MODEL_PATH=app/models/epoch50.pt
MODEL_CONFIDENCE=0.5
MODEL_IOU_THRESHOLD=0.45

# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=Vehicle Detection API
``` 