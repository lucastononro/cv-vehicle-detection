from typing import Dict, Type
from .base_model import BaseModel
from .yolo_model import YOLOModel
from .ocr_model import OCRModel
from pathlib import Path

class ModelFactory:
    def __init__(self):
        self._model_classes = {
            'yolo': YOLOModel,
            'ocr': OCRModel
        }
        self._loaded_models: Dict[str, BaseModel] = {}
    
    def register_model_class(self, model_type: str, model_class: Type[BaseModel]):
        """Register a new model class"""
        self._model_classes[model_type] = model_class
    
    def load_model(self, model_type: str, model_path: str, model_name: str) -> BaseModel:
        """Load a model of the specified type"""
        if model_type not in self._model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # If model is already loaded, return it
        if model_name in self._loaded_models:
            print(f"Model {model_name} already loaded")
            return self._loaded_models[model_name]
        
        try:
            model_class = self._model_classes[model_type]
            
            # Special handling for OCR model which doesn't need a model file
            if model_type == 'ocr':
                model = model_class(model_name=model_name)
            else:
                if not model_path:
                    raise ValueError(f"Model path required for {model_type} model")
                model = model_class(model_path, model_name)
            
            self._loaded_models[model_name] = model
            return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get a loaded model by name"""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model not loaded: {model_name}")
        return self._loaded_models[model_name]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all loaded models and their configurations"""
        return {
            name: {
                'type': type(model).__name__,
                'name': model.model_name
            }
            for name, model in self._loaded_models.items()
        } 