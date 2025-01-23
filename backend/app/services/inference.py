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
from .models.ocr_model import EasyOCRWrapper, TrOCRFinetunedWrapper, TesseractWrapper, TrOCRLargeWrapper, UnifiedOCRModel

class InferenceService:
    def __init__(self):
        self.model_factory = ModelFactory() #YOLO models only
        self.models_dir = Path("/app/app/models")
        
        try:
            # Load default YOLO model first
            model_path = self.models_dir / "yolo11n.pt"  # Docker path
            print(f"Loading YOLO model from {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
                
            self.default_model = self.model_factory.load_model('yolo', str(model_path), 'yolo11n')
            print("Successfully loaded YOLO model")
            
            # Initialize OCR model
            print("Initializing OCR model")
            self.ocr_model = None  # Will be set when needed
            print("Successfully initialized OCR model")
            
        except Exception as e:
            print(f"Failed to initialize models: {str(e)}")
            raise
    
    def get_model(self, model_name: Optional[str] = None) -> BaseModel:
        """Get a model by name, loading it if necessary"""
        if not model_name:
            return self.default_model
            
        try:
            # Try to get already loaded model
            return self.model_factory.get_model(model_name)
        except ValueError:
            # Model not loaded, try to load it
            model_path = self.models_dir / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            return self.model_factory.load_model('yolo', str(model_path), model_name)
    
    def get_ocr_model(self, model_name: Optional[str] = None) -> UnifiedOCRModel:
        """Get a UnifiedOCRModel by name, defaults to easyocr if none specified"""
        model_name = model_name or 'easyocr'  # Default to easyocr
        if not self.ocr_model or self.ocr_model.model_name != model_name:
            print(f"Creating new OCR model instance for: {model_name}")
            self.ocr_model = UnifiedOCRModel(model_name)
        return self.ocr_model

    def process_image(self, image_array: np.ndarray, model_name: Optional[str] = None, use_ocr: bool = True, ocr_model: Optional[str] = 'easyocr') -> Dict:
        """Process a single image and return detections"""
        try:
            print("\n=== Starting Image Processing ===")
            print(f"Model: {model_name}, OCR Enabled: {use_ocr}, OCR Model: {ocr_model}")
            
            # First get vehicle/license plate detections using YOLO
            print("Getting YOLO model...")
            model = self.get_model(model_name)
            print("Running YOLO prediction...")
            results = model.predict([image_array], batch_size=1, stream=False, draw_annotations=False)
            print("YOLO prediction complete")
            
            if not results or len(results) != 2:
                print("No valid results from YOLO")
                return {"model_name": model_name, "detections": [], "processed_frame": image_array.copy()}
            
            detections, _ = results
            print(f"Raw detections type: {type(detections)}")
            if detections:
                print(f"First detection type: {type(detections[0])}")
            
            # Ensure detections is a list of dictionaries
            if detections:
                if isinstance(detections[0], list):
                    print("Unwrapping nested detection list")
                    detections = detections[0]
                elif not isinstance(detections[0], dict):
                    print("Invalid detection format, resetting")
                    detections = []

            # Create a copy of the frame for annotations
            final_frame = image_array.copy()
            
            print(f"\nFound {len(detections)} total detections")

            # Get OCR model once if needed
            ocr_model_instance = None
            if use_ocr:
                print(f"Initializing OCR model: {ocr_model}")
                ocr_model_instance = self.get_ocr_model(ocr_model)
                print(f"Using OCR model: {ocr_model_instance.model_name}")

            for idx, detection in enumerate(detections):
                if not isinstance(detection, dict):
                    print(f"Skipping invalid detection {idx}")
                    continue
                    
                class_name = detection.get("class_name", "").lower()
                print(f"\nProcessing Detection {idx + 1}: Class = {class_name}")

                # Get bbox coordinates
                bbox = detection.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")

                # Draw detection box
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add class name with confidence
                class_conf = f"{detection.get('class_name', '')} ({detection.get('confidence', 0):.2f})"
                class_font_scale = 0.8
                class_thickness = 2
                class_size = cv2.getTextSize(class_conf, cv2.FONT_HERSHEY_SIMPLEX, class_font_scale, class_thickness)[0]

                # Draw class name background with padding
                padding = 5
                cv2.rectangle(final_frame, 
                            (x1, y1 - class_size[1] - padding * 2), 
                            (x1 + class_size[0] + padding * 2, y1),
                            (0, 255, 0), -1)  # Green background

                # Draw class name text in black
                cv2.putText(final_frame, class_conf,
                          (x1 + padding, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX,
                          class_font_scale, (0, 0, 0), class_thickness)

                # Add OCR if enabled and this is a license plate
                if use_ocr and ocr_model_instance and any(plate_term in class_name for plate_term in ["license", "plate", "licence", "number"]):
                    print("\n=== Starting OCR Processing ===")
                    try:
                        # Ensure valid crop region
                        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > image_array.shape[1] or y2 > image_array.shape[0]:
                            print("Invalid crop region, skipping OCR")
                            continue

                        print("Cropping license plate region...")
                        cropped = image_array[y1:y2, x1:x2]
                        if cropped.size == 0:
                            print("Empty crop region, skipping OCR")
                            continue

                        print(f"Cropped region shape: {cropped.shape}")
                        print("Starting OCR text extraction...")
                        text, confidence = ocr_model_instance.read_text_from_region(cropped, [0, 0, x2-x1, y2-y1])
                        print("OCR text extraction complete")

                        if text:
                            text = text.strip().upper().replace(' ', '')
                            print(f"OCR Success - Text: {text}, Confidence: {confidence:.2f}")
                            
                            detection["text"] = text
                            detection["text_confidence"] = confidence

                            # Draw OCR results in green to match detection
                            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            text_with_conf = f"OCR: {text} ({confidence:.2f})"
                            text_font_scale = 0.8
                            text_thickness = 2
                            text_size = cv2.getTextSize(text_with_conf, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)[0]

                            # Draw OCR text background in green
                            cv2.rectangle(final_frame, 
                                        (x1, y1 - text_size[1] - class_size[1] - padding * 4), 
                                        (x1 + text_size[0] + padding * 2, y1 - class_size[1] - padding * 2),
                                        (0, 255, 0), -1)

                            # Draw OCR text in black
                            cv2.putText(final_frame, text_with_conf,
                                      (x1 + padding, y1 - class_size[1] - padding * 3), cv2.FONT_HERSHEY_SIMPLEX,
                                      text_font_scale, (0, 0, 0), text_thickness)
                        else:
                            print("No text detected in license plate region")
                    except Exception as e:
                        print(f"Error in OCR processing: {str(e)}")
                        continue
                    print("=== OCR Processing Complete ===\n")

            print("\n=== Image Processing Complete ===")
            result = {
                "model_name": model_name,
                "detections": detections if isinstance(detections, list) else [],
                "processed_frame": final_frame
            }
            return result
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {"model_name": model_name, "detections": [], "processed_frame": image_array.copy()}
    
    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16, use_ocr: bool = True, ocr_model: Optional[str] = 'easyocr') -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of video frames with optimized batch inference"""
        # Get detections from YOLO
        detections, processed_frames = self.default_model.predict(frames, batch_size=batch_size, stream=False, draw_annotations=False)
        
        # Ensure detections is a list of lists
        if detections and not isinstance(detections[0], list):
            detections = [[d] for d in detections]
        
        # Create copies of frames for annotations
        final_frames = [frame.copy() for frame in frames]
        
        if use_ocr:
            print(f"\n=== OCR Batch Processing Start ===")
            print(f"Processing batch of {len(frames)} frames")
            
            # Get OCR model once for the batch
            print(f"Initializing OCR model: {ocr_model}")
            ocr_model_instance = self.get_ocr_model(ocr_model)
            print(f"Using OCR model: {ocr_model_instance.model_name}")
            
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
                    cv2.rectangle(final_frames[frame_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)  # Thinner green box
                    
                    # Add class name with confidence
                    class_conf = f"{detection['class_name']} ({detection['confidence']:.2f})"
                    class_font_scale = 0.8  # Smaller font
                    class_thickness = 2  # Thinner text
                    class_size = cv2.getTextSize(class_conf, cv2.FONT_HERSHEY_SIMPLEX, class_font_scale, class_thickness)[0]
                    
                    # Draw class name background with padding
                    padding = 5
                    cv2.rectangle(final_frames[frame_idx], 
                                (x1, y1 - class_size[1] - padding * 2), 
                                (x1 + class_size[0] + padding * 2, y1),
                                (0, 255, 0), -1)  # Green background
                    
                    # Draw class name text
                    cv2.putText(final_frames[frame_idx], class_conf,
                              (x1 + padding, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX,
                              class_font_scale, (0, 0, 0), class_thickness)  # Black text
                    
                    # Check for license plate and add OCR
                    if any(plate_term in class_name for plate_term in ["license", "plate", "licence", "number"]):
                        print(f"License plate detected at bbox: {bbox}")
                        
                        # Debug: Print cropped region dimensions
                        crop_height = y2 - y1
                        crop_width = x2 - x1
                        print(f"Cropped region dimensions: {crop_width}x{crop_height}")
                        
                        # Read text from the license plate region
                        text, confidence = ocr_model_instance.read_text_from_region(frame[y1:y2, x1:x2], [0, 0, x2-x1, y2-y1])
                        
                        if text:
                            print(f"✓ OCR Success - Text: {text}, Confidence: {confidence:.2f}")
                            # Add OCR results to the detection
                            detection["text"] = text
                            detection["text_confidence"] = confidence
                            
                            # Draw yellow box for OCR detection
                            cv2.rectangle(final_frames[frame_idx], (x1, y1), (x2, y2), (0, 255, 255), 3)
                            
                            # Add OCR text with confidence
                            text_with_conf = f"OCR: {text} ({confidence:.2f})"
                            ocr_font_scale = 0.8  # Smaller font
                            ocr_thickness = 2  # Thinner text
                            text_size = cv2.getTextSize(text_with_conf, cv2.FONT_HERSHEY_SIMPLEX, ocr_font_scale, ocr_thickness)[0]
                            
                            # Draw OCR text background with padding
                            cv2.rectangle(final_frames[frame_idx], 
                                        (x1, y1 - text_size[1] - class_size[1] - padding * 4), 
                                        (x1 + text_size[0] + padding * 2, y1 - class_size[1] - padding * 2),
                                        (0, 255, 255), -1)  # Yellow background
                            
                            # Draw OCR text
                            cv2.putText(final_frames[frame_idx], text_with_conf,
                                      (x1 + padding, y1 - class_size[1] - padding * 3), cv2.FONT_HERSHEY_SIMPLEX,
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