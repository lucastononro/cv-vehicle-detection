from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def predict(self, image: Union[np.ndarray, List[np.ndarray]], batch_size: int = 1, 
                stream: bool = False, draw_annotations: bool = False) -> Union[List[Dict], Tuple[List[Dict], List[np.ndarray]]]:
        """Process a single image or batch of images and return detections"""
        pass
    
    @abstractmethod
    def predict_video(self, video_path: str, batch_size: int = 16, save_path: Optional[str] = None) -> List[Dict]:
        """Process video in batches"""
        pass
    
    @abstractmethod
    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of video frames"""
        pass
    
    @abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """Get model class names"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model"""
        pass
    
    def read_text_from_region(self, image: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], Optional[float]]:
        """Read text from a specific region of the image. Default implementation returns None."""
        return None, None 