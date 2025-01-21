# API

This directory contains the API endpoints and routers for the Vehicle Detection system.

## Structure

```
api/
└── v1/                     # API version 1
    ├── endpoints/          # API endpoint implementations
    │   ├── videos.py      # Video processing and inference endpoints
    │   └── health.py      # Health check endpoint
    └── __init__.py        # Router initialization
```

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the service health status.

### Video Processing

#### Upload Video
```
POST /api/v1/videos/upload/
```
Upload a video file for processing.

**Request:**
- Form data with video file

**Response:**
```json
{
    "video_name": "string",
    "status": "success"
}
```

#### List Videos
```
GET /api/v1/videos/list/
```
List all uploaded videos.

#### Stream Video Inference
```
GET /api/v1/videos/inference/{video_name}/stream
```
Stream real-time inference results for a video.

**Query Parameters:**
- `model_name`: Optional model to use for inference

#### Image Classification
```
POST /api/v1/videos/classify-image/
```
Classify a single image.

#### Available Models
```
GET /api/v1/videos/models
```
Get list of available detection models.

#### Save Inference Results
```
POST /api/v1/videos/inference/{video_name}/save
```
Save inference results for a video.

#### Stream Video
```
GET /api/v1/videos/stream/{video_name}
```
Stream a processed video.

## Adding New Endpoints

1. Create a new file in `v1/endpoints/`
2. Define your FastAPI router and endpoints
3. Include the router in `v1/__init__.py`

Example:
```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint():
    return {"message": "New endpoint"}
``` 