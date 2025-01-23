from typing import Dict, Optional, List
from .models.ocr_model import (
    EasyOCRWrapper, 
    TesseractWrapper, 
    TrOCRFinetunedWrapper,
    TrOCRLargeWrapper,
    FastPlateWrapper,
    UnifiedOCRModel
)
from ..core.config import AVAILABLE_OCR_MODELS

class OCRModelFactory:
    def __init__(self):
        self._model_classes = {
            'tesseract': TesseractWrapper,
            'easyocr': EasyOCRWrapper,
            'trocr-finetuned': TrOCRFinetunedWrapper,
            'trocr-large': TrOCRLargeWrapper,
            'fastplate': FastPlateWrapper
        }
        self._loaded_models: Dict[str, UnifiedOCRModel] = {}

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available OCR model names"""
        return AVAILABLE_OCR_MODELS.copy()

    def get_model(self, model_name: str) -> UnifiedOCRModel:
        """Get an OCR model by name, loading it if necessary"""
        model_name = model_name.lower()
        
        # Return cached model if available
        if model_name in self._loaded_models:
            print(f"Using cached {model_name} model")
            return self._loaded_models[model_name]
        
        # Get model class
        model_class = self._model_classes.get(model_name)
        if not model_class:
            raise ValueError(f"Unknown OCR model: {model_name}")
        
        try:
            print(f"Loading {model_name} model...")
            # Create new model instance
            model = UnifiedOCRModel(model_name)
            self._loaded_models[model_name] = model
            print(f"Successfully loaded {model_name} model")
            return model
        except Exception as e:
            print(f"Error loading OCR model {model_name}: {str(e)}")
            raise 