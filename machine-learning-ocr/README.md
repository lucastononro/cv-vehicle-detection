# OCR for license plates

## Project Structure

- **models.py**: Contains various OCR model wrappers, including EasyOCR, TrOCR (fine-tuned and raw large), and Tesseract. Each wrapper handles model initialization, image processing, and optional post-processing.

- **preprocessor.py**: Provides the `OCRPreprocessor` class, which handles image preprocessing tasks such as resizing, denoising, and deskewing to prepare images for OCR.

- **postprocessor.py**: Implements the `LicensePlatePostProcessor` class, which corrects common OCR errors and ensures the output matches valid Brazilian license plate formats - havent shown to be that useful for now

- **test_ocr_models.py**: A testing script that evaluates the performance of each OCR model on a set of images. It calculates accuracy metrics and logs detailed results.

- **requirements.txt**: Lists the Python dependencies required to run the project, including libraries like `torch`, `transformers`, `easyocr`, and `pytesseract`.

- **images/**: A directory where input images for testing are stored. Images should be in PNG format and named should be the ground-truth license plate number.

- **runs_logs/**: Stores logs of test runs, including accuracy metrics and detailed results.

- **debug_output/**: Contains intermediate images generated during preprocessing for debugging purposes.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd machine-learning-ocr
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Tesseract is installed and accessible:
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On Windows: Download the installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) (or just do it with pip lol ;D)

## Usage

1. Place your test images in the `images/` directory. Ensure they are in PNG format and their name correspond to the ground-trugh.

2. Run the test script to evaluate all OCR models:
   ```bash
   python test_ocr_models.py
   ```

3. Check the `runs_logs/` directory for detailed results and accuracy metrics.

## Models Included

- **EasyOCR**: A lightweight OCR model that runs on CPU.
- **TrOCR Fine-Tuned**: A fine-tuned version of the TrOCR model for printed text.
- **TrOCR Raw Large**: A large version of the TrOCR model for printed text.
- **Tesseract**: An open-source OCR engine with custom configurations for license plates.