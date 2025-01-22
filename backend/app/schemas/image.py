from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    text: Optional[str] = None
    text_confidence: Optional[float] = None

class ImageResponse(BaseModel):
    filename: str
    size: int
    last_modified: datetime
    thumbnail: Optional[str] = None  # base64 encoded thumbnail image

class ImageUploadResponse(BaseModel):
    message: str
    filename: str

class ImageInferenceResponse(BaseModel):
    model_name: str
    detections: List[Detection]
    processed_image: str  # base64 encoded image 