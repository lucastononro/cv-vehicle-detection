import cv2
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Tuple
import tempfile
import os
from pathlib import Path
import numpy as np
from .models.model_factory import ModelFactory

class InferenceService:
    def __init__(self):
        self.model_factory = ModelFactory()
        # Load default YOLO model
        model_paths = [
            Path("/app/app/models/yolo11n.pt"),  # Docker path
            Path("models/yolo11n.pt"),          # Local path
            Path("app/models/yolo11n.pt"),      # Alternative path
            Path("../models/yolo11n.pt"),       # Relative path
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                if model_path.exists():
                    self.default_model = self.model_factory.load_model('yolo', str(model_path), 'yolo11n')
                    print(f"Successfully loaded model from {model_path}")
                    model_loaded = True
                    break
            except Exception as e:
                print(f"Failed to load model from {model_path}: {str(e)}")
                continue
        
        if not model_loaded:
            raise FileNotFoundError(f"Model not found in any of the following locations: {[str(p) for p in model_paths]}")
    
    def process_image(self, image_array: np.ndarray, model_name: Optional[str] = None) -> Dict:
        """Process a single image and return detections"""
        model = self.model_factory.get_model(model_name) if model_name else self.default_model
        detections = model.predict(image_array)
        
        return {
            "model_name": model.model_name,
            "detections": detections
        }
    
    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of video frames with optimized batch inference"""
        return self.default_model.process_video_batch(frames, batch_size)
    
    async def process_video_file(self, video_path: str, output_path: str, model_name: Optional[str] = None) -> str:
        """Process entire video file and save results"""
        model = self.model_factory.get_model(model_name) if model_name else self.default_model
        # For now, we'll use the YOLO save functionality only with YOLO models
        if isinstance(model, self.model_factory._model_classes['yolo']):
            results = model.model(video_path, save=True)
        else:
            raise NotImplementedError(f"Saving video not implemented for model type {type(model)}")
        return output_path
    
    async def process_video_stream(self, video_path: str, model_name: Optional[str] = None, batch_size: int = 4) -> AsyncGenerator[str, None]:
        """Process video in batches and yield results in real-time"""
        model = self.model_factory.get_model(model_name) if model_name else self.default_model
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_indices = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Process any remaining frames
                    if frames:
                        try:
                            detections, processed_frames = model.process_video_batch(frames, len(frames))
                            for frame_idx, dets in zip(frame_indices, detections):
                                result = {
                                    "model_name": model.model_name,
                                    "frame_index": frame_idx,
                                    "detections": dets
                                }
                                yield f"data: {json.dumps(result)}\n\n"
                                await asyncio.sleep(0.001)
                        except Exception as e:
                            print(f"Error processing final batch: {str(e)}")
                    break
                
                frames.append(frame)
                frame_indices.append(len(frame_indices))
                
                # Process batch when we have enough frames
                if len(frames) == batch_size:
                    try:
                        # Process batch using the optimized batch method
                        detections, processed_frames = model.process_video_batch(frames, batch_size)
                        
                        # Yield each frame's results
                        for frame_idx, dets in zip(frame_indices, detections):
                            result = {
                                "model_name": model.model_name,
                                "frame_index": frame_idx,
                                "detections": dets
                            }
                            yield f"data: {json.dumps(result)}\n\n"
                            await asyncio.sleep(0.001)  # Small delay to prevent overwhelming the client
                        
                        frames = []
                        frame_indices = []
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                        # Clear the problematic batch and continue
                        frames = []
                        frame_indices = []
                        continue
        
        finally:
            cap.release()
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models and their configurations"""
        return self.model_factory.list_models() 