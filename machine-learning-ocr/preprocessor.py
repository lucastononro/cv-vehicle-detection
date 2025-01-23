import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple

class OCRPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (200, 50), debug_output_dir: str = "debug_output"):
        self.target_size = target_size
        self.debug_output_dir = debug_output_dir
        self.rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.min_ar = 4.0
        self.max_ar = 5.0
        self.keep = 5

    def save_debug_image(self, image: np.ndarray, step_name: str, timestamp: str) -> None:
        os.makedirs(self.debug_output_dir, exist_ok=True)
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        filename = f"{timestamp}_{step_name}.png"
        filepath = os.path.join(self.debug_output_dir, filename)
        cv2.imwrite(filepath, image)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def resize_image(self, image: np.ndarray, width: int = 600) -> np.ndarray:
        h, w = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        moments = cv2.moments(image)
        if abs(moments['mu02']) < 1e-2:
            return image
        skew = moments['mu11'] / moments['mu02']
        height, width = image.shape[:2]
        M = np.array([[1, skew, -0.5 * height * skew],
                     [0, 1, 0]], dtype=np.float32)
        deskewed = cv2.warpAffine(
            image, 
            M, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[255, 255, 255]
        )
        return deskewed

    def preprocess(self, image: np.ndarray, verbose: bool = False) -> np.ndarray:
        return image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if verbose:
            self.save_debug_image(image, "00_original", timestamp)
        
        gray = self.to_grayscale(image)
        gray = self.resize_image(gray, width=600)
        if verbose:
            self.save_debug_image(gray, "01_grayscale", timestamp)
        
        denoised = self.denoise_image(gray)
        if verbose:
            self.save_debug_image(denoised, "02_denoised", timestamp)
        
        deskewed = self.deskew_image(denoised)
        if verbose:
            self.save_debug_image(deskewed, "03_deskewed", timestamp)
        
        return deskewed 