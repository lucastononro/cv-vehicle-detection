import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm

from preprocessor import OCRPreprocessor
from models import EasyOCRWrapper, TrOCRFineTunedWrapper, TesseractWrapper, TrOCRRawLargeWrapper

class OCRModelTester:
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.preprocessor = OCRPreprocessor()
        self.results = {}
        self.setup_models()

    def setup_models(self):
        # Initialize EasyOCR
        print("Initializing EasyOCR...")
        self.easyocr_reader = EasyOCRWrapper()
        
        # Initialize TrOCR Fine-Tuned
        print("Initializing TrOCR Fine-Tuned...")
        self.trocr_finetuned_reader = TrOCRFineTunedWrapper()
        
        # Initialize TrOCR Raw Large
        print("Initializing TrOCR Raw Large...")
        self.trocr_raw_large_reader = TrOCRRawLargeWrapper()
        
        # Initialize Tesseract
        print("Initializing Tesseract...")
        self.tesseract_reader = TesseractWrapper()

    def load_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                path = os.path.join(self.image_dir, filename)
                # Get ground truth from filename (without extension)
                ground_truth = os.path.splitext(filename)[0]
                image = cv2.imread(path)
                images[ground_truth] = image
        return images

    def process_with_easyocr(self, image: np.ndarray) -> str:
        processed = self.preprocessor.preprocess(image)
        
        # Save the processed image temporarily
        temp_path = os.path.join(self.preprocessor.debug_output_dir, 'temp_easyocr.png')
        os.makedirs(self.preprocessor.debug_output_dir, exist_ok=True)
        cv2.imwrite(temp_path, processed)
        
        try:
            result = self.easyocr_reader(temp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_with_trocr_finetuned(self, image: np.ndarray) -> str:
        processed = self.preprocessor.preprocess(image)
        
        # Save the processed image temporarily
        temp_path = os.path.join(self.preprocessor.debug_output_dir, 'temp_trocr_finetuned.png')
        os.makedirs(self.preprocessor.debug_output_dir, exist_ok=True)
        cv2.imwrite(temp_path, processed)
        
        try:
            result = self.trocr_finetuned_reader(temp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_with_trocr_raw_large(self, image: np.ndarray) -> str:
        processed = self.preprocessor.preprocess(image)
        
        # Save the processed image temporarily
        temp_path = os.path.join(self.preprocessor.debug_output_dir, 'temp_trocr_raw_large.png')
        os.makedirs(self.preprocessor.debug_output_dir, exist_ok=True)
        cv2.imwrite(temp_path, processed)
        
        try:
            result = self.trocr_raw_large_reader(temp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_with_tesseract(self, image: np.ndarray) -> str:
        processed = self.preprocessor.preprocess(image)
        
        # Save the processed image temporarily
        temp_path = os.path.join(self.preprocessor.debug_output_dir, 'temp_tesseract.png')
        os.makedirs(self.preprocessor.debug_output_dir, exist_ok=True)
        cv2.imwrite(temp_path, processed)
        
        try:
            result = self.tesseract_reader(temp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def run_tests(self):
        images = self.load_images()
        results = {
            'EasyOCR': {},
            'TrOCR Fine-Tuned': {},
            'TrOCR Raw Large': {},
            'Tesseract': {},
        }

        print("\nTesting OCR models...")
        for ground_truth, image in tqdm(images.items()):
            # Test EasyOCR
            easyocr_result = self.process_with_easyocr(image)
            results['EasyOCR'][ground_truth] = easyocr_result
            
            # Test TrOCR Fine-Tuned
            trocr_finetuned_result = self.process_with_trocr_finetuned(image)
            results['TrOCR Fine-Tuned'][ground_truth] = trocr_finetuned_result
            
            # Test TrOCR Raw Large
            trocr_raw_large_result = self.process_with_trocr_raw_large(image)
            results['TrOCR Raw Large'][ground_truth] = trocr_raw_large_result
            
            # Test Tesseract
            tesseract_result = self.process_with_tesseract(image)
            results['Tesseract'][ground_truth] = tesseract_result

        self.results = results
        return results

    def calculate_metrics(self):
        metrics = {}
        for model_name, predictions in self.results.items():
            correct = 0
            total = len(predictions)
            
            for ground_truth, predicted in predictions.items():
                if ground_truth == predicted:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            metrics[model_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        
        return metrics

    def display_results(self):
        metrics = self.calculate_metrics()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\nResults Summary:")
        print("=" * 50)
        
        # Create runs_logs directory if it doesn't exist
        logs_dir = "runs_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare the log content
        log_content = ["Results Summary:", "=" * 50, ""]
        
        for model_name, model_metrics in metrics.items():
            summary = f"\n{model_name}:\n"
            summary += f"Accuracy: {model_metrics['accuracy']:.2%}\n"
            summary += f"Correct: {model_metrics['correct']}/{model_metrics['total']}"
            print(summary)
            log_content.extend(summary.split('\n'))
        
        # Create detailed results DataFrame
        results_data = []
        for ground_truth in next(iter(self.results.values())).keys():
            row = {'Ground Truth': ground_truth}
            for model_name in self.results.keys():
                row[model_name] = self.results[model_name][ground_truth]
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        print("\nDetailed Results:")
        print(df)
        
        # Add DataFrame to log content
        log_content.extend(["\nDetailed Results:", df.to_string()])
        
        # Save to file
        log_file = os.path.join(logs_dir, f"results_{timestamp}.txt")
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_content))
        
        print(f"\nResults saved to: {log_file}")
        return df

def main():
    tester = OCRModelTester('images')
    tester.run_tests()
    tester.display_results()

if __name__ == "__main__":
    main() 