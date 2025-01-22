import os
import easyocr
from PIL import Image
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline
)
import pytesseract
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Tuple, Optional
from postprocessor import LicensePlatePostProcessor

class CharacterSimilarity:
    """Character similarity mappings for OCR correction"""
    # Similar looking characters (bidirectional mapping)
    SIMILAR_CHARS = {
        'B': '8',
        'D': '0',
        'O': '0',
        'Q': '0',
        'S': '5',
        'Z': '2',
        'I': '1',
        'A': '4',
        'G': '6',
        'T': '7',
    }
    
    # Create reverse mappings
    REVERSE_SIMILAR = {v: k for k, v in SIMILAR_CHARS.items()}
    
    @classmethod
    def get_similar_letter(cls, digit: str) -> Optional[str]:
        """Convert a digit to a similar looking letter"""
        return cls.REVERSE_SIMILAR.get(digit)
    
    @classmethod
    def get_similar_digit(cls, letter: str) -> Optional[str]:
        """Convert a letter to a similar looking digit"""
        return cls.SIMILAR_CHARS.get(letter)

class EasyOCRWrapper:
    def __init__(self, use_postprocessing: bool = True):
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False)  # CPU only
        self.post_processor = LicensePlatePostProcessor() if use_postprocessing else None
        print("OCR model loaded successfully!")
        
    def __call__(self, image_path: str) -> str:
        """Run OCR on an image."""
        try:
            print(f"\nProcessing image: {image_path}")
            
            # Load and verify image
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Run OCR
            print("Running OCR...")
            results = self.reader.readtext(image_path)
            
            # Extract text
            text = ' '.join([result[1] for result in results])
            text = text.strip().upper()
            print(f"Raw OCR Result: {text}")
            
            # Post-process if enabled
            if self.post_processor:
                processed = self.post_processor.process(text)
                if processed:
                    print(f"Post-processed Result: {processed}")
                    return processed
                else:
                    print("Post-processing failed, returning raw result")
            
            return text
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

class TrOCRFineTunedWrapper:
    def __init__(self, model_name="DunnBC22/trocr-base-printed_license_plates_ocr", 
                 processor_name="microsoft/trocr-base-printed",
                 use_postprocessing: bool = True):
        print(f"Loading TrOCR Fine-Tuned model: {model_name}...")
        self.model_name = model_name
        
        # Initialize processor and model
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Ensure CPU usage
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize post-processor
        self.post_processor = LicensePlatePostProcessor() if use_postprocessing else None
        
        print("TrOCR Fine-Tuned model loaded successfully!")
        
    def process_image(self, image: Image.Image) -> str:
        """Process a single image."""
        # Prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip().upper()
        
    def __call__(self, image_path: str) -> str:
        """Run OCR on an image."""
        try:
            print(f"\nProcessing image with TrOCR Fine-Tuned: {image_path}")
            
            # Load and verify image
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            print("Running OCR...")
            text = self.process_image(image)
            print(f"Raw TrOCR Fine-Tuned Result: {text}")
            
            # Post-process if enabled
            if self.post_processor:
                processed = self.post_processor.process(text)
                if processed:
                    print(f"Post-processed Result: {processed}")
                    return processed
                else:
                    print("Post-processing failed, returning raw result")
            
            return text
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame, processor: TrOCRProcessor, max_target_length: int = 128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name + text
        file_name = self.df['file_name'].iloc[idx]  # Use iloc for proper indexing
        text = self.df['text'].iloc[idx]
        
        # Prepare image
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Add labels by encoding text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids
        
        # Make sure PAD tokens are ignored by loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

class TesseractWrapper:
    """Wrapper for Tesseract OCR"""
    def __init__(self, use_postprocessing: bool = True):
        """
        Initialize Tesseract OCR wrapper.
        Args:
            use_postprocessing: Whether to use license plate post-processing
        """
        print("Initializing Tesseract OCR...")
        
        # Configure Tesseract for license plates
        self.config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Custom OCR settings
        self.custom_config = {
            'lang': 'eng',  # English language
            'config': self.config,  # Custom configuration
            'nice': 0,  # Priority (0 = highest)
            'timeout': 10  # Timeout in seconds
        }
        
        # Initialize post-processor
        self.post_processor = LicensePlatePostProcessor() if use_postprocessing else None
        
        # Test Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR initialized successfully!")
        except Exception as e:
            print("Warning: Tesseract installation not found or not working properly!")
            print(f"Error: {str(e)}")
            print("Please ensure Tesseract is installed and accessible.")
            print("On macOS: brew install tesseract")
            print("On Ubuntu: sudo apt-get install tesseract-ocr")
            print("On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki")
        
    def preprocess_for_tesseract(self, image: Image.Image) -> Image.Image:
        """Apply Tesseract-specific preprocessing"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to a reasonable size if too small or too large
        width, height = image.size
        target_width = 300  # Good size for license plates
        if width != target_width:
            ratio = target_width / width
            new_size = (target_width, int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
        
    def __call__(self, image_path: str) -> str:
        """Run OCR on an image."""
        try:
            print(f"\nProcessing image with Tesseract: {image_path}")
            
            # Load and verify image
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_for_tesseract(image)
            
            # Run OCR with custom configuration
            print("Running OCR...")
            text = pytesseract.image_to_string(
                processed_image,
                **self.custom_config
            )
            
            # Clean up text
            text = text.strip().upper()
            print(f"Raw Tesseract Result: {text}")
            
            # Post-process if enabled
            if self.post_processor:
                processed = self.post_processor.process(text)
                if processed:
                    print(f"Post-processed Result: {processed}")
                    return processed
                else:
                    print("Post-processing failed, returning raw result")
            
            return text
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

class TrOCRRawLargeWrapper:
    def __init__(self, model_name="microsoft/trocr-large-printed", 
                 processor_name="microsoft/trocr-large-printed",
                 use_postprocessing: bool = True):
        print(f"Loading TrOCR Raw Large model: {model_name}...")
        self.model_name = model_name
        
        # Initialize processor and model
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Ensure CPU usage
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize post-processor
        self.post_processor = LicensePlatePostProcessor() if use_postprocessing else None
        
        print("TrOCR Raw Large model loaded successfully!")
        
    def process_image(self, image: Image.Image) -> str:
        """Process a single image."""
        # Prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip().upper()
        
    def __call__(self, image_path: str) -> str:
        """Run OCR on an image."""
        try:
            print(f"\nProcessing image with TrOCR Raw Large: {image_path}")
            
            # Load and verify image
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            print("Running OCR...")
            text = self.process_image(image)
            print(f"Raw TrOCR Raw Large Result: {text}")
            
            # Post-process if enabled
            if self.post_processor:
                processed = self.post_processor.process(text)
                if processed:
                    print(f"Post-processed Result: {processed}")
                    return processed
                else:
                    print("Post-processing failed, returning raw result")
            
            return text
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return "" 