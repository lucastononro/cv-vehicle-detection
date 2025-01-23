import easyocr
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel
import cv2
import os
from datetime import datetime
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import pytesseract
from fast_plate_ocr import ONNXPlateRecognizer
import tempfile

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

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to the image"""
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image using moments method"""
        # Calculate moments
        moments = cv2.moments(image)
        
        if abs(moments['mu02']) < 1e-2:
            # Return original image if skew is negligible
            return image
        
        # Calculate skew angle
        skew = moments['mu11'] / moments['mu02']
        height, width = image.shape[:2]
        
        # Create transformation matrix as numpy array
        M = np.array([[1, skew, -0.5 * height * skew],
                     [0, 1, 0]], dtype=np.float32)
        
        # Apply affine transform
        deskewed = cv2.warpAffine(
            image, 
            M, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[255, 255, 255]
        )
        return deskewed

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
            self.save_debug_image(gray, "01_grayscale", timestamp)
        
        # # 2. Apply blackhat operation to enhance dark text
        # blackhat = self.apply_blackhat(gray)
        # if verbose:
        #     self.save_debug_image(blackhat, "02_blackhat", timestamp)
        
        # 2. Apply denoising to clean up the image
        denoised = self.denoise_image(gray)
        if verbose:
            self.save_debug_image(denoised, "02_denoised", timestamp)
        
        # 3. Apply deskewing
        deskewed = self.deskew_image(denoised)
        if verbose:
            self.save_debug_image(deskewed, "03_deskewed", timestamp)
        
        return deskewed

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

class EasyOCRWrapper:
    """Wrapper class for EasyOCR model implementation"""
    _reader = None  # Class-level cache for the reader
    
    def __init__(self, languages: List[str] = ['en']):
        if EasyOCRWrapper._reader is None:
            print("Initializing EasyOCR reader (first time only)...")
            try:
                EasyOCRWrapper._reader = easyocr.Reader(
                    languages,
                    gpu=False,  # Force CPU mode for stability
                    download_enabled=False,  # Prevent automatic downloads
                    verbose=False
                )
                print("EasyOCR reader initialized successfully")
            except Exception as e:
                print(f"Error initializing EasyOCR: {str(e)}")
                raise
        self.reader = EasyOCRWrapper._reader
        
    def extract_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extract text from image using EasyOCR"""
        try:
            print("Starting EasyOCR text detection...")
            print(f"Input image shape: {image.shape}")
            
            # Convert image to grayscale if it's color
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Resize if image is too small
            min_size = 100
            if min(gray.shape) < min_size:
                scale = min_size / min(gray.shape)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)
            
            print("Running OCR...")
            results = self.reader.readtext(gray)
            print(f"OCR complete. Found {len(results)} results")
            
            if not results:
                print("No text detected")
                return []
                
            # Extract text and confidence
            processed_results = []
            for result in results:
                text = result[1].strip().upper()
                conf = result[2]
                if text:  # Only include non-empty text
                    processed_results.append((text, conf))
                    print(f"Detected: {text} (conf: {conf:.2f})")
            
            return processed_results
            
        except Exception as e:
            print(f"Error in EasyOCR text extraction: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

class TrOCRFinetunedWrapper:
    def __init__(self, model_path):
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

class TesseractWrapper:
    """Wrapper class for Tesseract OCR engine."""
    
    def __init__(self, language: str = 'eng'):
        self.language = language
        
    def extract_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract text from an image using Tesseract OCR.
        
        Args:
            image: numpy array of the image
            
        Returns:
            List of tuples containing (text, confidence_score)
        """
        try:
            print("Starting Tesseract text detection...")
            print(f"Input image shape: {image.shape}")
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # Convert BGR to RGB for PIL
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image)
            
            # Get text and confidence data
            data = pytesseract.image_to_data(pil_image, lang=self.language, output_type=pytesseract.Output.DICT)
            
            # Process results
            results = []
            for i, text in enumerate(data['text']):
                conf = float(data['conf'][i])
                if conf > 0 and text.strip():  # Only include non-empty text with valid confidence
                    text = text.strip().upper()
                    conf = conf / 100.0  # Convert confidence to 0-1 range
                    results.append((text, conf))
                    print(f"Detected: {text} (conf: {conf:.2f})")
            
            print(f"Tesseract complete. Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in Tesseract text extraction: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

class TrOCRLargeWrapper:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        
    def process_image(self, image: Image) -> str:
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text 

class FastPlateWrapper:
    """Wrapper class for FastPlate OCR implementation"""
    def __init__(self, model_name: str = 'argentinian-plates-cnn-model'):
        print("Initializing FastPlate OCR...")
        try:
            self.model = ONNXPlateRecognizer(model_name)
            print("FastPlate OCR initialized successfully")
        except Exception as e:
            print(f"Error initializing FastPlate OCR: {str(e)}")
            raise

    def extract_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract text from an image using FastPlate OCR.
        
        Args:
            image: numpy array of the image
            
        Returns:
            List of tuples containing (text, confidence_score)
        """
        try:
            print("Starting FastPlate text detection...")
            print(f"Input image shape: {image.shape}")
            
            # Save image temporarily since FastPlate requires a file path
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_file:
                cv2.imwrite(temp_file.name, image)
                
                # Run OCR
                print("Running OCR...")
                predictions = self.model.run(temp_file.name)
                
                # Process results
                results = []
                if predictions:
                    # FastPlate doesn't provide confidence scores, so we'll use 1.0
                    # Take first prediction if it's a list
                    text = predictions[0] if isinstance(predictions, list) else str(predictions)
                    text = text.strip().upper()
                    if text:
                        results.append((text, 1.0))
                        print(f"Detected: {text} (conf: 1.00)")
                
                print(f"FastPlate complete. Found {len(results)} results")
                return results
                
        except Exception as e:
            print(f"Error in FastPlate text extraction: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

class OCRModelFactory:
    def __init__(self):
        self._model_classes = {
            'easyocr': EasyOCRWrapper,
            'trocr_finetuned': TrOCRFinetunedWrapper,
            'tesseract': TesseractWrapper,
            'trocr_large': TrOCRLargeWrapper,
            'fastplate': FastPlateWrapper
        }
        self._loaded_models = {}  # Cache for loaded models

    def get_model(self, model_name: str) -> Optional[object]:
        """Get an OCR model by name, loading it only if needed"""
        print(f"Requesting OCR model: {model_name}")
        
        if model_name not in self._model_classes:
            raise ValueError(f"OCR model '{model_name}' not found")
            
        # Return cached model if available
        if model_name in self._loaded_models:
            print(f"Using cached {model_name} model")
            return self._loaded_models[model_name]
            
        # Load the requested model
        print(f"Loading {model_name} model...")
        if model_name == 'easyocr':
            model = self._model_classes[model_name](['en'])
        elif model_name == 'trocr_finetuned':
            model = self._model_classes[model_name]('microsoft/trocr-base-printed')
        elif model_name == 'tesseract':
            model = self._model_classes[model_name]()
        elif model_name == 'trocr_large':
            model = self._model_classes[model_name]()
        elif model_name == 'fastplate':
            model = self._model_classes[model_name]()
        
        # Cache the model
        self._loaded_models[model_name] = model
        print(f"Successfully loaded {model_name} model")
        return model

class UnifiedOCRModel(BaseModel):
    def __init__(self, model_name: str):
        print(f"\n=== Initializing UnifiedOCRModel ===")
        print(f"Requested model: {model_name}")
        self._model_name = model_name
        try:
            print("Creating OCR model factory...")
            factory = OCRModelFactory()
            print(f"Getting model {model_name} from factory...")
            self.model = factory.get_model(model_name)
            print("OCR model initialization complete")
        except Exception as e:
            print(f"Error initializing OCR model: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_class_names(self) -> List[str]:
        return ["text"]

    def read_text_from_region(self, image: np.ndarray, bbox: List[float], verbose: bool = False) -> Tuple[Optional[str], Optional[float]]:
        print(f"\n=== OCR read_text_from_region ===")
        print(f"Image shape: {image.shape}")
        print(f"Bbox: {bbox}")
        
        try:
            if hasattr(self.model, 'extract_text'):
                print("Using extract_text method")
                results = self.model.extract_text(image)
                print(f"Got {len(results)} results from OCR")
                
                if results:
                    # Get the result with highest confidence
                    text, conf = max(results, key=lambda x: x[1])
                    print(f"Best result: {text} (conf: {conf:.2f})")
                    return text, conf
                else:
                    print("No text detected")
                    return None, None
            else:
                print("Model does not have extract_text method")
                return None, None
        except Exception as e:
            print(f"Error in read_text_from_region: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, None

    def predict(self, images: List[np.ndarray], batch_size: int = 1, stream: bool = False, draw_annotations: bool = True) -> Tuple[List[Dict], List[np.ndarray]]:
        results = []
        processed_frames = []

        for image in images:
            text, _ = self.read_text_from_region(image, [0, 0, image.shape[1], image.shape[0]])
            results.append([{'text': text}])
            processed_frames.append(image)

        return results, processed_frames 

    def predict_video(self, video_path: str, save: bool = False) -> None:
        """Not implemented for UnifiedOCRModel"""
        raise NotImplementedError("Video prediction not implemented for UnifiedOCRModel")

    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16) -> Tuple[List[Dict], List[np.ndarray]]:
        """Not meant to be used directly for UnifiedOCRModel"""
        return self.predict(frames, batch_size=batch_size, stream=False, draw_annotations=True) 