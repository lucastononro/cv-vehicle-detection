from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
import base64
from datetime import datetime
from ....services.label_generator import LabelGeneratorService
from ....repositories.s3 import S3Repository
from pydantic import BaseModel

router = APIRouter()
s3_repo = S3Repository()
label_generator = LabelGeneratorService()

# Response models
class LabelInfo(BaseModel):
    text: str
    confidence: float
    bbox: List[float]
    image_name: str

class LabelGenerationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    labels: Optional[List[LabelInfo]] = None
    processed_image: Optional[str] = None

class StoredLabel(BaseModel):
    image_name: str
    text: str
    confidence: float
    bbox: List[float]
    cropped_image: str  # base64 encoded image
    created_at: datetime

@router.get("/labels/", response_model=List[StoredLabel])
async def list_labels():
    """List all generated labels from the labels bucket"""
    try:
        # Get all files from the labels bucket
        files = s3_repo.list_files(bucket_type="labels")
        labels = []
        
        # Process only JSON files that contain label information
        for file in files:
            if file['Key'].endswith('.json'):
                # Get label data
                label_data = s3_repo.get_file_bytes(file['Key'], bucket_type="labels")
                if label_data:
                    import json
                    label_info = json.loads(label_data.decode('utf-8'))
                    
                    # Get corresponding cropped image
                    image_key = file['Key'].replace('.json', '.jpg')
                    image_data = s3_repo.get_file_bytes(image_key, bucket_type="labels")
                    
                    if image_data:
                        # Create response object
                        labels.append(StoredLabel(
                            image_name=label_info['image_name'],
                            text=label_info['text'],
                            confidence=label_info['confidence'],
                            bbox=label_info['bbox'],
                            cropped_image=f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
                            created_at=file['LastModified']
                        ))
        
        # Sort by creation date, newest first
        labels.sort(key=lambda x: x.created_at, reverse=True)
        return labels
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inference/{image_name}", response_model=LabelGenerationResponse)
async def generate_labels(image_name: str):
    """Generate labels for an image using the label generator service"""
    try:
        # Get image from S3
        image_bytes = s3_repo.get_file_bytes(image_name, bucket_type="images")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process image with label generator
        result = label_generator.process_image_for_labels(image, image_name)
        
        if not result['success']:
            return LabelGenerationResponse(
                success=False,
                message=result.get('message', 'Failed to generate labels')
            )
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', result['processed_frame'])
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return LabelGenerationResponse(
            success=True,
            labels=result.get('labels', []),
            processed_image=f"data:image/jpeg;base64,{img_base64}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/labels/{label_text}")
async def delete_label(label_text: str):
    """Delete a label and its associated files from the labels bucket"""
    try:
        # List all files in labels bucket
        files = s3_repo.list_files(bucket_type="labels")
        
        # Find and delete matching files
        deleted = False
        for file in files:
            if label_text in file['Key']:
                s3_repo.s3_client.delete_object(
                    Bucket=s3_repo.buckets["labels"],
                    Key=file['Key']
                )
                deleted = True
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Label not found")
            
        return JSONResponse(content={"message": f"Label {label_text} deleted successfully"})
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 