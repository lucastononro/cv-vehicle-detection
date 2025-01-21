from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional

class VideoResponse(BaseModel):
    filename: str
    size: int
    last_modified: datetime
    thumbnail: Optional[str] = None

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class ImageClassificationResponse(BaseModel):
    filename: str
    detections: List[Detection]

class VideoUploadResponse(BaseModel):
    message: str
    filename: str

class VideoInferenceResponse(BaseModel):
    message: str
    original_video: str
    processed_video: str 