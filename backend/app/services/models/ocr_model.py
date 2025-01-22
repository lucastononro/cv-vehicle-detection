import easyocr
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel
import cv2

class OCRModel(BaseModel):
    def __init__(self, model_path: Optional[str] = None, model_name: str = "easyocr"):
        self._model_name = model_name
        # Initialize EasyOCR reader with English language
        self.reader = easyocr.Reader(['en'])
    
    @property
    def model_name(self) -> str:
        return self._model_name

    def get_class_names(self) -> List[str]:
        return ["text"]
    
    def read_text_from_region(self, image: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], Optional[float]]:
        """Read text from a specific region of the image"""
        try:
            # Extract coordinates and ensure they're within image bounds
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            print(f"\nOCR Debug Info:")
            print(f"Original image shape: {image.shape}")
            print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                print("Error: Cropped region is empty")
                return None, None
            
            print(f"Cropped region shape: {cropped.shape}")
            print(f"Cropped region dtype: {cropped.dtype}")
            print(f"Cropped region min/max values: {np.min(cropped)}/{np.max(cropped)}")
            
            # Perform OCR on cropped region
            detections = self.reader.readtext(cropped)
            if not detections:
                print("No text detected in region")
                return None, None
            
            # Get the text with highest confidence
            best_detection = max(detections, key=lambda x: x[2])
            text = best_detection[1]
            confidence = best_detection[2]
            
            # Clean up the text (remove spaces, convert to uppercase)
            text = text.upper().replace(' ', '')
            
            return text, confidence
        except Exception as e:
            print(f"Error reading text from region: {str(e)}")
            return None, None
        
    def predict(self, images: List[np.ndarray], batch_size: int = 1, stream: bool = False, draw_annotations: bool = True) -> Tuple[List[Dict], List[np.ndarray]]:
        """This method is maintained for compatibility but should not be used directly.
        Use read_text_from_region instead."""
        results = []
        processed_frames = []
        
        for image in images:
            # Just return empty results since this method shouldn't be used directly
            results.append([])
            processed_frames.append(image)
                
        return results, processed_frames

    def predict_video(self, video_path: str, save: bool = False) -> None:
        """Not implemented for OCR model"""
        raise NotImplementedError("Video prediction not implemented for OCR model")

    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16) -> Tuple[List[Dict], List[np.ndarray]]:
        """Not meant to be used directly for OCR"""
        return self.predict(frames, batch_size=batch_size, stream=False, draw_annotations=True) 