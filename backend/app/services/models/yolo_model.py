from .base_model import BaseModel
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import cv2

class YOLOModel(BaseModel):
    @staticmethod
    def draw_text_with_background(img: np.ndarray, text: str, pos: Tuple[int, int], 
                                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 5.0,
                                text_color: Tuple[int, int, int] = (255, 255, 255),
                                bg_color: Tuple[int, int, int] = (0, 100, 0),
                                thickness: int = 4):
        """Helper function to draw text with background"""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle
        padding = 15
        bg_rect = [
            pos[0],
            pos[1] - text_height - padding,
            text_width + 2*padding,
            text_height + 2*padding
        ]
        
        # Draw background rectangle
        cv2.rectangle(img, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(img, text, pos, font, font_scale, text_color, thickness)

    def __init__(self, model_path: Union[str, Path], model_name: str):
        self._model_name = model_name
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            print(f"Loading YOLO model {model_name} from {model_path}")
            self.model = YOLO(str(model_path))
            self._class_names = self.model.names
            print(f"Loaded {model_name} with classes:", self._class_names)
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            raise
    
    def predict(self, image: Union[np.ndarray, List[np.ndarray]], batch_size: int = 1, stream: bool = False, 
               draw_annotations: bool = False) -> Union[List[Dict], Tuple[List[Dict], List[np.ndarray]]]:
        """Process a single image or batch of images and return detections"""
        # Convert single image to list if needed
        if isinstance(image, np.ndarray):
            images_to_process = [image]
        else:
            images_to_process = image

        # Process images as a batch
        results = self.model(images_to_process, stream=stream, batch=batch_size)
        
        all_detections = []
        processed_images = []
        
        for idx, result in enumerate(results):
            frame_detections = []
            if result.boxes:  # Check if there are any detections
                boxes = result.boxes.data.cpu().numpy()
                
                # Draw annotations if requested
                if draw_annotations:
                    img = images_to_process[idx].copy()
                    class_counts = {}  # Counter for each class
                    
                    # First pass: count objects
                    for box in boxes:
                        class_id = int(box[5])
                        class_name = self._class_names[class_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Draw total count and class counts
                    y_offset = 60  # Start higher
                    total_objects = sum(class_counts.values())
                    self.draw_text_with_background(img, f"Total Objects: {total_objects}", (20, y_offset), 
                                                 bg_color=(0, 100, 0), font_scale=2.0,
                                                 text_color=(255, 255, 255))
                    y_offset += 60
                    
                    # Draw individual class counts
                    for class_name, count in sorted(class_counts.items()):
                        counter_text = f"{class_name}: {count}"
                        self.draw_text_with_background(img, counter_text, (20, y_offset), 
                                                     bg_color=(0, 100, 0), font_scale=2.0,
                                                     text_color=(255, 255, 255))
                        y_offset += 60
                    
                    # Second pass: draw bounding boxes and labels
                    for box in boxes:
                        class_id = int(box[5])
                        class_name = self._class_names[class_id]
                        confidence = float(box[4])
                        bbox = box[:4].astype(int)
                        
                        # Draw thicker bounding box
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
                        
                        # Draw label with background
                        label = f"{class_name} ({confidence:.2f})"
                        self.draw_text_with_background(img, label, (bbox[0], bbox[1] - 30), 
                                                     bg_color=(0, 100, 0), font_scale=1.5)
                    
                    processed_images.append(img)
                else:
                    processed_images.append(images_to_process[idx].copy())
                
                for box in boxes:
                    class_id = int(box[5])
                    frame_detections.append({
                        "bbox": [float(x) for x in box[:4].tolist()],  # x1, y1, x2, y2
                        "confidence": float(box[4]),  # confidence score
                        "class_id": class_id,
                        "class_name": self._class_names[class_id]
                    })
            else:
                # No detections, add empty list and original image
                frame_detections = []
                if draw_annotations:
                    processed_images.append(images_to_process[idx].copy())
            
            all_detections.append(frame_detections)
        
        if draw_annotations:
            return all_detections, processed_images
        return all_detections
    
    def predict_video_stream(self, frame: np.ndarray, draw_annotations: bool = True) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Process a single frame from video stream with annotations"""
        if draw_annotations:
            detections, processed_frames = self.predict([frame], batch_size=1, stream=False, draw_annotations=True)
            return detections[0], processed_frames[0] if processed_frames else None
        else:
            detections = self.predict([frame], batch_size=1, stream=False, draw_annotations=False)
            return detections[0], None

    def process_video_batch(self, frames: List[np.ndarray], batch_size: int = 16) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of video frames with optimized batch inference"""
        detections, processed_frames = self.predict(frames, batch_size=batch_size, stream=True, draw_annotations=True)
        return detections, processed_frames if processed_frames else frames

    def predict_video(self, video_path: str, batch_size: int = 16, save_path: Optional[str] = None) -> List[Dict]:
        """Process video in batches for better GPU utilization"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_indices = []
        detections_by_frame = []
        current_frame = 0
        
        # Initialize video writer if save_path is provided
        writer = None
        if save_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_indices.append(current_frame)
                current_frame += 1
                
                # Process batch when we have enough frames
                if len(frames) == batch_size:
                    batch_detections, processed_frames = self.process_video_batch(frames, batch_size)
                    for idx, (dets, proc_frame) in enumerate(zip(batch_detections, processed_frames)):
                        detections_by_frame.append((frame_indices[idx], dets))
                        if writer:
                            writer.write(proc_frame)
                    frames = []
                    frame_indices = []
            
            # Process remaining frames
            if frames:
                batch_detections, processed_frames = self.process_video_batch(frames, len(frames))
                for idx, (dets, proc_frame) in enumerate(zip(batch_detections, processed_frames)):
                    detections_by_frame.append((frame_indices[idx], dets))
                    if writer:
                        writer.write(proc_frame)
            
            return sorted(detections_by_frame, key=lambda x: x[0])
            
        finally:
            cap.release()
            if writer:
                writer.release()
    
    def get_class_names(self) -> Dict[int, str]:
        return self._class_names
    
    @property
    def model_name(self) -> str:
        return self._model_name 