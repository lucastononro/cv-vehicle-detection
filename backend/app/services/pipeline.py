from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

class ModelPipeline:
    def __init__(self):
        self.models_dir = "/app/app/models"
        
        # Initialize models
        vehicle_model_path = f"{self.models_dir}/yolo11n.pt"
        plate_model_path = f"{self.models_dir}/license_plate_detector.pt"
        
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.ocr_model = easyocr.Reader(['en'])
        
        # Define available models
        self.models = {
            'yolo11n': self.vehicle_model,
            'license_plate_detector': self.plate_model,
            'easyocr': self.ocr_model
        }
        
        # Define default pipeline
        self.default_pipeline = ['yolo11n', 'license_plate_detector', 'easyocr']

    def crop_detection(self, image: np.ndarray, box: List[float]) -> np.ndarray:
        """Crop image based on detection box"""
        x1, y1, x2, y2 = map(int, box)
        return image[y1:y2, x1:x2]

    def draw_detection(self, image: np.ndarray, box: List[float], label: str, color: Tuple[int, int, int]) -> None:
        """Draw detection box and label on image"""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Add label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        padding = 5
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 2*padding), (x1 + text_width + 2*padding, y1), color, -1)
        cv2.putText(image, label, (x1 + padding, y1 - padding), font, font_scale, (255, 255, 255), thickness)

    def process_image(self, image: np.ndarray, use_ocr: bool = True, pipeline_steps: Optional[List[str]] = None) -> Dict:
        """Process image through pipeline"""
        # Make copy of image for drawing
        final_image = image.copy()
        all_detections = []
        
        # Use provided pipeline steps or default
        steps = pipeline_steps if pipeline_steps else self.default_pipeline
        if use_ocr is False and 'easyocr' in steps:
            steps.remove('easyocr')
            
        # Process each step
        if 'yolo11n' in steps:
            # Detect vehicles
            vehicle_results = self.vehicle_model.predict(image, verbose=False)[0]
            for detection in vehicle_results.boxes.data.tolist():
                box = detection[:4]
                conf = detection[4]
                cls = int(detection[5])
                
                # Draw vehicle detection
                label = f"Vehicle {conf:.2f}"
                self.draw_detection(final_image, box, label, (0, 255, 0))  # Green
                
                if 'license_plate_detector' in steps:
                    # Crop and detect license plate
                    vehicle_crop = self.crop_detection(image, box)
                    plate_results = self.plate_model.predict(vehicle_crop, verbose=False)[0]
                    
                    for plate_detection in plate_results.boxes.data.tolist():
                        plate_box = plate_detection[:4]
                        plate_conf = plate_detection[4]
                        
                        # Adjust plate coordinates to original image
                        adjusted_box = [
                            plate_box[0] + box[0],
                            plate_box[1] + box[1],
                            plate_box[2] + box[0],
                            plate_box[3] + box[1]
                        ]
                        
                        # Draw plate detection
                        plate_label = f"Plate {plate_conf:.2f}"
                        self.draw_detection(final_image, adjusted_box, plate_label, (255, 0, 0))  # Blue
                        
                        if 'easyocr' in steps:
                            # Crop and read plate text
                            plate_crop = self.crop_detection(image, adjusted_box)
                            ocr_results = self.ocr_model.readtext(plate_crop)
                            
                            for ocr_result in ocr_results:
                                text = ocr_result[1]
                                ocr_conf = ocr_result[2]
                                
                                # Draw OCR result
                                ocr_label = f"OCR: {text} {ocr_conf:.2f}"
                                self.draw_detection(final_image, adjusted_box, ocr_label, (0, 255, 255))  # Yellow
                                
                                all_detections.append({
                                    'type': 'ocr',
                                    'text': text,
                                    'confidence': ocr_conf,
                                    'box': adjusted_box
                                })
                        
                        all_detections.append({
                            'type': 'plate',
                            'confidence': plate_conf,
                            'box': adjusted_box
                        })
                
                all_detections.append({
                    'type': 'vehicle',
                    'confidence': conf,
                    'class': cls,
                    'box': box
                })

        return {
            'image': final_image,
            'detections': all_detections
        } 