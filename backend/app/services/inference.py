import cv2
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Tuple
import tempfile
import os
from pathlib import Path
import numpy as np
from .models.model_factory import ModelFactory
from .models.base_model import BaseModel

class InferenceService:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.models_dir = Path("/app/app/models")
        
        try:
            # Load default YOLO model first
            model_path = self.models_dir / "yolo11n.pt"  # Docker path
            print(f"Loading YOLO model from {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
                
            self.default_model = self.model_factory.load_model('yolo', str(model_path), 'yolo11n')
            print("Successfully loaded YOLO model")
            
            # Initialize OCR model separately (no model file needed)
            print("Initializing OCR model")
            self.ocr_model = self.model_factory.load_model('ocr', "", 'easyocr')
            print("Successfully initialized OCR model")
            
        except Exception as e:
            print(f"Failed to initialize models: {str(e)}")
            raise
    
    def get_model(self, model_name: Optional[str] = None) -> BaseModel:
        """Get a model by name, loading it if necessary"""
        if not model_name:
            return self.default_model
            
        # Special case for OCR
        if model_name == 'easyocr':
            return self.ocr_model
            
        try:
            # Try to get already loaded model
            return self.model_factory.get_model(model_name)
        except ValueError:
            # Model not loaded, try to load it
            model_path = self.models_dir / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            return self.model_factory.load_model('yolo', str(model_path), model_name)
    
    def process_image(self, image_array: np.ndarray, model_name: Optional[str] = None, use_ocr: bool = True) -> Dict:
        """Process a single image and return detections"""
        # First get vehicle/license plate detections
        model = self.get_model(model_name)
        detections, _ = model.predict([image_array], batch_size=1, stream=False, draw_annotations=False)  # We'll handle drawing ourselves
        
        # Create a copy of the frame for annotations
        final_frame = image_array.copy()
        
        # Process each detection and draw bounding boxes
        print("\n=== Processing Detections ===")
        print(f"Found {len(detections[0])} total detections")
        
        for idx, detection in enumerate(detections[0]):
            class_name = detection.get("class_name", "").lower()
            print(f"\nDetection {idx + 1}: Class = {class_name} (original: {detection.get('class_name')})")
            
            # Get bbox coordinates
            bbox = detection["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw detection box
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box for all detections
            
            # Add class name with confidence
            class_conf = f"{detection['class_name']} ({detection['confidence']:.2f})"
            class_font_scale = 1.2
            class_thickness = 3
            class_size = cv2.getTextSize(class_conf, cv2.FONT_HERSHEY_SIMPLEX, class_font_scale, class_thickness)[0]
            
            # Draw class name background
            cv2.rectangle(final_frame, 
                        (x1, y1 - class_size[1] - 20), 
                        (x1 + class_size[0] + 10, y1 - 10),
                        (0, 255, 0), -1)  # Green background
            
            # Draw class name text
            cv2.putText(final_frame, class_conf,
                      (x1 + 5, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                      class_font_scale, (0, 0, 0), class_thickness)  # Black text
            
            # Add OCR if enabled and this is a license plate
            if use_ocr and any(plate_term in class_name for plate_term in ["license", "plate", "licence", "number"]):
                print(f"License plate detected at bbox: {bbox}")
                
                # Debug: Print cropped region dimensions
                crop_height = y2 - y1
                crop_width = x2 - x1
                print(f"Cropped region dimensions: {crop_width}x{crop_height}")
                
                # Read text from the license plate region
                text, confidence = self.ocr_model.read_text_from_region(image_array, bbox)
                
                if text:
                    print(f"✓ OCR Success - Text: {text}, Confidence: {confidence:.2f}")
                    # Add OCR results to the detection
                    detection["text"] = text
                    detection["text_confidence"] = confidence
                    
                    # Draw yellow box for OCR detection
                    cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                    
                    # Add OCR text with confidence
                    text_with_conf = f"OCR: {text} ({confidence:.2f})"
                    ocr_font_scale = 1.2
                    ocr_thickness = 3
                    text_size = cv2.getTextSize(text_with_conf, cv2.FONT_HERSHEY_SIMPLEX, ocr_font_scale, ocr_thickness)[0]
                    
                    # Draw OCR text background - position it above the class name
                    cv2.rectangle(final_frame, 
                                (x1, y1 - text_size[1] - class_size[1] - 30), 
                                (x1 + text_size[0] + 10, y1 - class_size[1] - 20),
                                (0, 255, 255), -1)  # Yellow background
                    
                    # Draw OCR text
                    cv2.putText(final_frame, text_with_conf,
                              (x1 + 5, y1 - class_size[1] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                              ocr_font_scale, (0, 0, 0), ocr_thickness)  # Black text
                else:
                    print("✗ No text detected in license plate region")
        
        result = {
            "model_name": model.model_name,
            "detections": detections[0],
            "processed_frame": final_frame
        }
        return result
    
    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16, use_ocr: bool = True) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of video frames with optimized batch inference"""
        # Get detections from YOLO
        detections, processed_frames = self.default_model.predict(frames, batch_size=batch_size, stream=False, draw_annotations=False)
        
        # Create copies of frames for annotations
        final_frames = [frame.copy() for frame in frames]
        
        if use_ocr:
            print(f"\n=== OCR Batch Processing Start ===")
            print(f"Processing batch of {len(frames)} frames")
            
            # Process each frame's detections with OCR
            for frame_idx, (frame_dets, frame) in enumerate(zip(detections, frames)):
                print(f"\nFrame {frame_idx + 1}/{len(frames)}:")
                print(f"Found {len(frame_dets)} detections")
                
                for det_idx, detection in enumerate(frame_dets):
                    class_name = detection.get("class_name", "").lower()
                    print(f"Detection {det_idx + 1}: Class = {class_name}")
                    
                    # Get bbox coordinates
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Draw detection box
                    cv2.rectangle(final_frames[frame_idx], (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box for all detections
                    
                    # Add class name with confidence
                    class_conf = f"{detection['class_name']} ({detection['confidence']:.2f})"
                    class_font_scale = 1.2
                    class_thickness = 3
                    class_size = cv2.getTextSize(class_conf, cv2.FONT_HERSHEY_SIMPLEX, class_font_scale, class_thickness)[0]
                    
                    # Draw class name background
                    cv2.rectangle(final_frames[frame_idx], 
                                (x1, y1 - class_size[1] - 20), 
                                (x1 + class_size[0] + 10, y1 - 10),
                                (0, 255, 0), -1)  # Green background
                    
                    # Draw class name text
                    cv2.putText(final_frames[frame_idx], class_conf,
                              (x1 + 5, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                              class_font_scale, (0, 0, 0), class_thickness)  # Black text
                    
                    # Check for license plate and add OCR
                    if any(plate_term in class_name for plate_term in ["license", "plate", "licence", "number"]):
                        print(f"License plate detected at bbox: {bbox}")
                        
                        # Debug: Print cropped region dimensions
                        crop_height = y2 - y1
                        crop_width = x2 - x1
                        print(f"Cropped region dimensions: {crop_width}x{crop_height}")
                        
                        # Read text from the license plate region
                        text, confidence = self.ocr_model.read_text_from_region(frame, bbox)
                        
                        if text:
                            print(f"✓ OCR Success - Text: {text}, Confidence: {confidence:.2f}")
                            # Add OCR results to the detection
                            detection["text"] = text
                            detection["text_confidence"] = confidence
                            
                            # Draw yellow box for OCR detection
                            cv2.rectangle(final_frames[frame_idx], (x1, y1), (x2, y2), (0, 255, 255), 4)
                            
                            # Add OCR text with confidence
                            text_with_conf = f"OCR: {text} ({confidence:.2f})"
                            ocr_font_scale = 1.2
                            ocr_thickness = 3
                            text_size = cv2.getTextSize(text_with_conf, cv2.FONT_HERSHEY_SIMPLEX, ocr_font_scale, ocr_thickness)[0]
                            
                            # Draw OCR text background - position it above the class name
                            cv2.rectangle(final_frames[frame_idx], 
                                        (x1, y1 - text_size[1] - class_size[1] - 30), 
                                        (x1 + text_size[0] + 10, y1 - class_size[1] - 20),
                                        (0, 255, 255), -1)  # Yellow background
                            
                            # Draw OCR text
                            cv2.putText(final_frames[frame_idx], text_with_conf,
                                      (x1 + 5, y1 - class_size[1] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                      ocr_font_scale, (0, 0, 0), ocr_thickness)  # Black text
                        else:
                            print("✗ No text detected in license plate region")
            print("\n=== OCR Batch Processing Complete ===\n")
        
        return detections, final_frames
    
    async def process_video_file(self, video_path: str, output_path: str, model_name: Optional[str] = None) -> str:
        """Process entire video file and save results"""
        model = self.get_model(model_name)
        # For now, we'll use the YOLO save functionality only with YOLO models
        if isinstance(model, self.model_factory._model_classes['yolo']):
            results = model.model(video_path, save=True)
        else:
            raise NotImplementedError(f"Saving video not implemented for model type {type(model)}")
        return output_path
    
    async def process_video_stream(self, video_path: str, model_name: Optional[str] = None, batch_size: int = 4) -> AsyncGenerator[str, None]:
        """Process video in batches and yield results in real-time"""
        model = self.get_model(model_name)
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

    def get_available_models(self) -> List[str]:
        """List all available .pt model files in the models directory"""
        try:
            return [f.name for f in self.models_dir.glob("*.pt")]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return [] 