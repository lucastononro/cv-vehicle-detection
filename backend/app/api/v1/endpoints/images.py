from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import tempfile

from ....services.inference import InferenceService
from ....repositories.s3 import S3Repository
from ....schemas.image import ImageResponse, ImageUploadResponse, ImageInferenceResponse
from ....services.pipeline import ModelPipeline

router = APIRouter()
s3_repo = S3Repository()
inference_service = InferenceService()
pipeline_service = ModelPipeline()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def generate_thumbnail(image: np.ndarray) -> Optional[bytes]:
    """Generate a compressed thumbnail from an image"""
    try:
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Calculate new dimensions (max 360px)
        max_size = 360
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        
        # Resize image
        thumbnail = cv2.resize(image, (new_width, new_height))
        
        # Encode with JPEG compression
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
        return None

def process_image_for_storage(image_path: str) -> tuple[Optional[bytes], Optional[bytes]]:
    """Process image before storage and generate thumbnail"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        # Generate thumbnail first
        thumbnail_data = generate_thumbnail(image)
        
        # Resize if image is too large (max dimension 1920px)
        height, width = image.shape[:2]
        max_dimension = 1920
        if height > max_dimension or width > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            image = cv2.resize(image, (new_width, new_height))
        
        # Encode main image with optimal quality
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), thumbnail_data
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

@router.post("/upload/", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file to S3"""
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate content type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save file temporarily for processing
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as temp_image:
            # Save uploaded file content
            content = await file.read()
            with open(temp_image.name, 'wb') as f:
                f.write(content)
            
            # Process image
            image = cv2.imread(temp_image.name)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Encode image with optimal quality
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_image = buffer.tobytes()
            
            # Use original filename
            filename = file.filename
            
            # Upload processed image to S3
            s3_repo.upload_bytes(processed_image, filename, bucket_type="images")
            
            # Use the same image for thumbnail
            thumbnail_key = f"thumbnails/{os.path.splitext(filename)[0]}.jpg"
            s3_repo.upload_bytes(processed_image, thumbnail_key, bucket_type="thumbnails")
            
            return ImageUploadResponse(
                message="Image uploaded successfully",
                filename=filename
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/", response_model=List[ImageResponse])
async def list_images():
    """List all uploaded images"""
    try:
        files = s3_repo.list_files(bucket_type="images")
        image_list = []
        
        for obj in files:
            # Get thumbnail (which is the same as the original image)
            thumbnail_key = f"thumbnails/{os.path.splitext(obj['Key'])[0]}.jpg"
            thumbnail = None
            try:
                thumbnail_data = s3_repo.get_file_bytes(thumbnail_key, bucket_type="thumbnails")
                if thumbnail_data:
                    thumbnail = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode('utf-8')}"
            except Exception:
                pass
            
            image_list.append(
                ImageResponse(
                    filename=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    thumbnail=thumbnail
                )
            )
        
        return image_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inference/{image_name}", response_model=ImageInferenceResponse)
async def get_inference(
    image_name: str,
    model_name: Optional[str] = None,
    use_ocr: bool = True
):
    """Get inference results for an image"""
    try:
        # Download image from S3 images bucket
        image_bytes = s3_repo.get_file_bytes(image_name, bucket_type="images")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process image
        result = inference_service.process_image(image, model_name, use_ocr=use_ocr)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', result['processed_frame'])
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Return results with proper response model
        return ImageInferenceResponse(
            model_name=result['model_name'],
            detections=result['detections'],
            processed_image=f"data:image/jpeg;base64,{img_base64}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_name}")
async def get_image(image_name: str):
    """Get original image"""
    try:
        # Get image bytes from S3
        image_bytes = s3_repo.get_file_bytes(image_name, bucket_type="images")
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Return the image bytes directly
        return Response(content=image_bytes, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/{image_name}", response_model=ImageInferenceResponse)
async def get_pipeline_inference(
    image_name: str,
    use_ocr: bool = True,
    pipeline: str = None,
    s3_service: S3Service = Depends(get_s3_service),
    pipeline_service: ModelPipeline = Depends(get_pipeline_service),
):
    """
    Get inference results for an image using model pipeline
    """
    try:
        # Get image from S3
        image_bytes = await s3_service.get_file(image_name)
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Process pipeline steps
        pipeline_steps = pipeline.split(',') if pipeline else None
        result = pipeline_service.process_image(image, use_ocr=use_ocr, pipeline_steps=pipeline_steps)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', result['image'])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        processed_image = f"data:image/jpeg;base64,{processed_image}"

        return {
            "detections": result['detections'],
            "processed_image": processed_image
        }

    except Exception as e:
        logger.error(f"Error in pipeline inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 