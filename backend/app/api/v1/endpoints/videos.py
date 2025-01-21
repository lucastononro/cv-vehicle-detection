from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
import os
from ....core.config import settings
from ....schemas.video import VideoResponse, ImageClassificationResponse, VideoUploadResponse, VideoInferenceResponse, Detection
from ultralytics import YOLO
import cv2
import tempfile
from ....repositories.s3 import S3Repository
from ....services.inference import InferenceService
import numpy as np
import base64
import asyncio

router = APIRouter()
s3_repo = S3Repository()
inference_service = InferenceService()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=os.getenv('S3_ENDPOINT_URL')
)

BUCKET_NAME = "videos"
THUMBNAIL_BUCKET = "thumbnails"
MODEL_PATH = "../../models/yolo11n.pt"

# Ensure buckets exist
try:
    s3_repo.s3_client.head_bucket(Bucket=BUCKET_NAME)
    s3_repo.s3_client.head_bucket(Bucket=THUMBNAIL_BUCKET)
except ClientError:
    s3_repo.s3_client.create_bucket(Bucket=BUCKET_NAME)
    s3_repo.s3_client.create_bucket(Bucket=THUMBNAIL_BUCKET)

def generate_thumbnail(video_path: str) -> Optional[bytes]:
    """Generate a thumbnail from a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Resize frame to thumbnail size
            height, width = frame.shape[:2]
            max_size = 360
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            thumbnail = cv2.resize(frame, (new_width, new_height))
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        return None
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

@router.post("/classify-image/", response_model=ImageClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """Classify a single image and return detections"""
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png, or bmp)")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        result = inference_service.process_image(image)
        
        return ImageClassificationResponse(
            filename=file.filename,
            detections=[
                Detection(
                    class_id=det.get('class_id'),
                    class_name=det.get('class_name'),
                    confidence=det.get('confidence'),
                    bbox=det.get('bbox')
                ) for det in result['detections']
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save video temporarily to generate thumbnail
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=True) as temp_video:
            # Save uploaded file content
            content = await file.read()
            with open(temp_video.name, 'wb') as f:
                f.write(content)
            
            # Generate and upload thumbnail
            thumbnail_data = generate_thumbnail(temp_video.name)
            if thumbnail_data:
                thumbnail_key = f"thumbnails/{os.path.splitext(file.filename)[0]}.jpg"
                s3_repo.upload_bytes(thumbnail_data, thumbnail_key)
            
            # Upload video
            with open(temp_video.name, 'rb') as f:
                s3_repo.upload_file(f, file.filename)
            
        return VideoUploadResponse(
            message="Video uploaded successfully",
            filename=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/", response_model=List[VideoResponse])
async def list_videos():
    try:
        files = s3_repo.list_files()
        video_list = []
        
        for obj in files:
            if obj['Key'].startswith('thumbnails/'):
                continue
                
            # Try to get thumbnail
            thumbnail_key = f"thumbnails/{os.path.splitext(obj['Key'])[0]}.jpg"
            thumbnail = None
            try:
                thumbnail_data = s3_repo.get_file_bytes(thumbnail_key)
                if thumbnail_data:
                    thumbnail = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode('utf-8')}"
            except Exception:
                pass
            
            video_list.append(
                VideoResponse(
                    filename=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    thumbnail=thumbnail
                )
            )
        
        return video_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inference/{video_name}/stream")
async def stream_inference(video_name: str, request: Request, background_tasks: BackgroundTasks, model_name: Optional[str] = None):
    """Stream video with real-time inference"""
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_name)[1], delete=False) as temp_video:
            # Download video from S3
            s3_repo.download_file(video_name, temp_video.name)
            
            # Process and stream the video
            cap = cv2.VideoCapture(temp_video.name)
            
            try:
                async def generate_frames():
                    try:
                        while cap.isOpened():
                            # Check if client is still connected
                            if await request.is_disconnected():
                                print("Client disconnected, stopping video processing")
                                break

                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process frame with YOLO with annotations enabled
                            result = inference_service.process_image(frame, model_name)
                            
                            # Get the processed frame with annotations
                            processed_frame = result.get('processed_frame', frame)
                            
                            # Encode frame to JPEG
                            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame_bytes = buffer.tobytes()
                            
                            # Yield frame with multipart boundary
                            yield (
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                            )
                            
                            # Use a shorter sleep to maintain responsiveness
                            await asyncio.sleep(0.001)
                    except Exception as e:
                        print(f"Error in frame generation: {str(e)}")
                    finally:
                        cap.release()
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_video.name)
                        except Exception as e:
                            print(f"Error cleaning up temp file: {str(e)}")

                # Create a response with cleanup
                return StreamingResponse(
                    generate_frames(),
                    media_type="multipart/x-mixed-replace;boundary=frame"
                )
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                if 'cap' in locals():
                    cap.release()
                raise e
            
    except Exception as e:
        print(f"Error in stream_inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available model files"""
    return inference_service.get_available_models()

@router.post("/inference/{video_name}/save", response_model=VideoInferenceResponse)
async def save_inference(video_name: str):
    """Process video and save results back to S3"""
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_name)[1], delete=False) as temp_video:
            # Download video from S3
            s3_repo.download_file(video_name, temp_video.name)
            
            # Process video and save results
            output_path = f"results_{video_name}"
            await inference_service.process_video_file(temp_video.name, output_path)
            
            return VideoInferenceResponse(
                message="Video processed successfully",
                original_video=video_name,
                processed_video=output_path
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream/{video_name}")
async def stream_video(video_name: str):
    """Stream video without inference"""
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_name)[1], delete=False) as temp_video:
            # Download video from S3
            s3_repo.download_file(video_name, temp_video.name)
            
            def iterfile():
                with open(temp_video.name, mode="rb") as file_like:
                    yield from file_like
                os.unlink(temp_video.name)  # Delete the temp file after streaming
            
            return StreamingResponse(
                iterfile(),
                media_type="video/mp4"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 