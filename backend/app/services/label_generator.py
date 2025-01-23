from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
import json
from .inference import InferenceService
from .models.base_model import BaseModel
from .models.ocr_model import GPT4VisionWrapper, UnifiedOCRModel

class LabelGeneratorService(InferenceService):
    def __init__(self):
        super().__init__()
        self.models_dir = Path("/app/app/models")
        
        try:
            # Load only the license plate detector model
            model_path = self.models_dir / "license_plate_detector.pt"
            print(f"Loading license plate detector model from {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"License plate detector model not found at {model_path}")
                
            self.default_model = self.model_factory.load_model('yolo', str(model_path), 'license_plate_detector')
            print("Successfully loaded license plate detector model")
            
            # Initialize GPT-4 Vision OCR model
            print("Initializing GPT-4 Vision OCR model")
            self.ocr_model = UnifiedOCRModel('gpt4-vision')
            print("Successfully initialized GPT-4 Vision OCR model")
            
        except Exception as e:
            print(f"Failed to initialize models: {str(e)}")
            raise

    def process_image_for_labels(self, image_array: np.ndarray, image_name: str) -> Dict:
        """Process a single image and generate labels for detected license plates"""
        try:
            print("\n=== Starting Label Generation Process ===")
            
            # Get license plate detections using YOLO
            print("Running license plate detection...")
            results = self.default_model.predict([image_array], batch_size=1, stream=False, draw_annotations=False)
            
            if not results or len(results) != 2:
                print("No valid results from detector")
                return {
                    "success": False,
                    "message": "No detections found",
                    "processed_frame": image_array.copy()
                }
            
            detections, _ = results
            
            # Ensure detections is a list
            if isinstance(detections, dict):
                detections = [detections]
            elif isinstance(detections[0], list):
                detections = detections[0]
            
            print(f"\nFound {len(detections)} license plate detections")
            
            # Create a copy of the frame for annotations
            final_frame = image_array.copy()
            generated_labels = []
            
            for idx, detection in enumerate(detections):
                if not isinstance(detection, dict):
                    print(f"Skipping invalid detection {idx}")
                    continue
                
                # Get bbox coordinates
                bbox = detection.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Ensure valid crop region
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > image_array.shape[1] or y2 > image_array.shape[0]:
                    print(f"Invalid crop region for detection {idx}")
                    continue
                
                # Crop license plate region
                cropped = image_array[y1:y2, x1:x2]
                if cropped.size == 0:
                    print(f"Empty crop region for detection {idx}")
                    continue
                
                print(f"\nProcessing detection {idx + 1}:")
                print(f"Crop dimensions: {cropped.shape}")
                
                # Get OCR text using GPT-4 Vision
                text, confidence = self.ocr_model.read_text_from_region(cropped, [0, 0, x2-x1, y2-y1])
                
                if text:
                    text = text.strip().upper().replace(' ', '')
                    print(f"✓ OCR Success - Text: {text}, Confidence: {confidence:.2f}")
                    
                    # Store detection info
                    label_info = {
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "image_name": image_name
                    }
                    generated_labels.append(label_info)
                    
                    # Draw annotations
                    cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add text overlay
                    text_with_conf = f"{text} ({confidence:.2f})"
                    font_scale = 0.8
                    thickness = 2
                    text_size = cv2.getTextSize(text_with_conf, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    
                    # Draw text background
                    cv2.rectangle(final_frame, 
                                (x1, y1 - text_size[1] - 10),
                                (x1 + text_size[0] + 10, y1),
                                (0, 255, 0), -1)
                    
                    # Draw text
                    cv2.putText(final_frame, text_with_conf,
                              (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale, (0, 0, 0), thickness)
                    
                    # Save cropped plate and label info
                    self._save_label_data(cropped, label_info)
                else:
                    print("✗ No text detected in license plate region")
            
            print("\n=== Label Generation Complete ===")
            return {
                "success": True,
                "labels": generated_labels,
                "processed_frame": final_frame
            }
            
        except Exception as e:
            print(f"Error in label generation: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": str(e),
                "processed_frame": image_array.copy()
            }

    def _save_label_data(self, cropped_image: np.ndarray, label_info: Dict) -> None:
        """Save the cropped image and its label information to S3"""
        try:
            from ..repositories.s3 import S3Repository
            s3_repo = S3Repository()
            
            # Generate unique filename for the cropped image
            base_name = Path(label_info["image_name"]).stem
            plate_text = label_info["text"]
            crop_filename = f"{base_name}_{plate_text}.jpg"
            label_filename = f"{base_name}_{plate_text}.json"
            
            # Encode cropped image
            _, img_encoded = cv2.imencode('.jpg', cropped_image)
            
            # Save cropped image to S3
            s3_repo.upload_bytes(img_encoded.tobytes(), crop_filename, "labels")
            
            # Save label information to S3
            label_json = json.dumps(label_info)
            s3_repo.upload_bytes(label_json.encode(), label_filename, "labels")
            
            print(f"Saved label data for {plate_text}")
            
        except Exception as e:
            print(f"Error saving label data: {str(e)}") 