from typing import Dict, Type
from .base_model import BaseModel
from .yolo_model import YOLOModel
from pathlib import Path

class ModelFactory:
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._model_classes = {
            'yolo': YOLOModel,
        }
    
    def register_model_class(self, model_type: str, model_class: Type[BaseModel]):
        """Register a new model class"""
        self._model_classes[model_type] = model_class
    
    def load_model(self, model_type: str, model_path: str, model_name: str) -> BaseModel:
        """Load a model of the specified type"""
        if model_type not in self._model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_name in self._models:
            return self._models[model_name]
        
        model_class = self._model_classes[model_type]
        model = model_class(model_path, model_name)
        self._models[model_name] = model
        return model
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get a loaded model by name"""
        if model_name not in self._models:
            raise ValueError(f"Model not loaded: {model_name}")
        return self._models[model_name]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all loaded models and their class names"""
        return {
            name: {
                "type": model.__class__.__name__,
                "classes": model.get_class_names()
            }
            for name, model in self._models.items()
        } 