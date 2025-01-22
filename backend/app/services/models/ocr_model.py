import easyocr
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel
import cv2
import os
from datetime import datetime

try:
    from skimage.segmentation import clear_border
except ImportError:
    # Fallback implementation if skimage is not available
    def clear_border(image):
        return image

class OCRPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (200, 50), debug_output_dir: str = "debug_output"):
        self.target_size = target_size
        self.debug_output_dir = debug_output_dir
        # Initialize kernels for morphological operations
        self.rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Parameters for license plate detection
        self.min_ar = 4.0
        self.max_ar = 5.0
        self.keep = 5  # Number of contours to keep

    def save_debug_image(self, image: np.ndarray, step_name: str, timestamp: str) -> None:
        """Save intermediate processing step image"""
        # Create debug directory if it doesn't exist
        os.makedirs(self.debug_output_dir, exist_ok=True)
        
        # Ensure image is in uint8 format
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Save image
        filename = f"{timestamp}_{step_name}.png"
        filepath = os.path.join(self.debug_output_dir, filename)
        cv2.imwrite(filepath, image)
        print(f"Saved debug image: {filepath}")

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def resize_image(self, image: np.ndarray, width: int = 600) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def apply_blackhat(self, gray: np.ndarray) -> np.ndarray:
        """Apply blackhat morphological operation to emphasize dark text on light background"""
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rect_kernel)

    def find_light_regions(self, gray: np.ndarray) -> np.ndarray:
        """Find light regions using morphological closing and Otsu's threshold"""
        # Apply morphological closing
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.square_kernel)
        # Apply Otsu's thresholding
        _, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return light

    def compute_gradient(self, blackhat: np.ndarray) -> np.ndarray:
        """Compute Sobel gradient in x-direction"""
        # Compute Sobel gradient
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        
        # Normalize gradient
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        return grad_x.astype(np.uint8)

    def process_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Process gradient with blur and morphological operations"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gradient, (5, 5), 0)
        # Apply morphological closing
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, self.rect_kernel)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh

    def find_plate_region(self, thresh: np.ndarray, light_mask: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find license plate region using contours"""
        # Combine threshold with light regions
        thresh = cv2.bitwise_and(thresh, thresh, mask=light_mask)
        thresh = cv2.dilate(thresh, self.cleanup_kernel, iterations=2)
        thresh = cv2.erode(thresh, self.cleanup_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.keep]
        
        # Initialize license plate ROI
        lp_roi = None
        binary_roi = None
        
        # Loop through contours
        for c in contours:
            # Get bounding box and compute aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            
            # Check if aspect ratio matches license plate
            if self.min_ar <= ar <= self.max_ar:
                # Extract license plate region
                lp_roi = gray[y:y+h, x:x+w]
                # Create binary ROI
                _, binary_roi = cv2.threshold(
                    lp_roi, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )
                # Remove border artifacts
                if binary_roi is not None:
                    # Create a mask slightly smaller than the image
                    mask = np.zeros_like(binary_roi)
                    h, w = binary_roi.shape
                    margin = 2
                    mask[margin:-margin, margin:-margin] = 1
                    # Apply mask
                    binary_roi = binary_roi * mask
                break
        
        return lp_roi, binary_roi

    def preprocess(self, image: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Apply full preprocessing pipeline"""
        # Create timestamp for this processing run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if verbose:
            self.save_debug_image(image, "00_original", timestamp)
        
        # 1. Convert to grayscale and resize
        gray = self.to_grayscale(image)
        gray = self.resize_image(gray, width=600)
        if verbose:
            self.save_debug_image(gray, "01_grayscale", timestamp)        # 2. Apply blackhat operation
        blackhat = self.apply_blackhat(gray)
        if verbose:
            self.save_debug_image(blackhat, "02_blackhat", timestamp)
        return blackhat
        
        # # 3. Find light regions
        # light = self.find_light_regions(gray)
        # if verbose:
        #     self.save_debug_image(light, "03_light_regions", timestamp)
        
        # # 4. Compute gradient
        # gradient = self.compute_gradient(blackhat)
        # if verbose:
        #     self.save_debug_image(gradient, "04_gradient", timestamp)
        
        # # 5. Process gradient
        # thresh = self.process_gradient(gradient)
        # if verbose:
        #     self.save_debug_image(thresh, "05_thresh", timestamp)
        
        # # 6. Find plate region
        # lp_roi, binary_roi = self.find_plate_region(thresh, light, gray)
        
        # if lp_roi is None:
        #     return gray  # Return original grayscale if no plate found
        
        # if verbose:
        #     if lp_roi is not None:
        #         self.save_debug_image(lp_roi, "06_plate_roi", timestamp)
        #     if binary_roi is not None:
        #         self.save_debug_image(binary_roi, "07_binary_roi", timestamp)
        
        # return binary_roi if binary_roi is not None else lp_roi

class OCRModel(BaseModel):
    def __init__(self, model_path: Optional[str] = None, model_name: str = "easyocr", debug_output_dir: str = "debug_output"):
        self._model_name = model_name
        # Initialize EasyOCR reader with English language
        self.reader = easyocr.Reader(['en'], gpu=True)
        # Initialize preprocessor
        self.preprocessor = OCRPreprocessor(debug_output_dir=debug_output_dir)
    
    @property
    def model_name(self) -> str:
        return self._model_name

    def get_class_names(self) -> List[str]:
        return ["text"]
    
    def read_text_from_region(self, image: np.ndarray, bbox: List[float], verbose: bool = False) -> Tuple[Optional[str], Optional[float]]:
        """Read text from a specific region of the image"""
        try:
            # Extract coordinates and ensure they're within image bounds
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if verbose:
                print(f"\nOCR Debug Info:")
                print(f"Original image shape: {image.shape}")
                print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                if verbose:
                    print("Error: Cropped region is empty")
                return None, None
            
            if verbose:
                print(f"Cropped region shape: {cropped.shape}")
                print(f"Cropped region dtype: {cropped.dtype}")
                print(f"Cropped region min/max values: {np.min(cropped)}/{np.max(cropped)}")
            
            # Apply preprocessing pipeline with verbose flag
            processed = self.preprocessor.preprocess(cropped, verbose=verbose)
            
            # Perform OCR on processed region
            detections = self.reader.readtext(processed)
            if not detections:
                if verbose:
                    print("No text detected in region")
                return None, None
            
            # Get the text with highest confidence
            best_detection = max(detections, key=lambda x: x[2])
            text = best_detection[1]
            confidence = best_detection[2]
            
            # Clean up the text (remove spaces, convert to uppercase)
            text = text.upper().replace(' ', '')
            
            if verbose:
                print(f"Detected text: {text}")
                print(f"Confidence: {confidence:.2f}")
            
            return text, confidence
        except Exception as e:
            if verbose:
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