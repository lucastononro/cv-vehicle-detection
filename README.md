# cv-vehicle-detection

##  DISCLAIMER: This is not a production-ready application. It is a proof of concept.

# What is this repo?

This is a toy project created to implement vehicle detection, license plate detection and OCR composing what we call ALPR (Automated License Plate Recognition).

# What was carried out?

1. First I started looking for custom training of Yolo models using YOLOv11.
2. A lot of data could be found on roboflow so I designed a simple training pipeline for YoloV11 for custom models focused on license plate detections
3. Then we trained and monitored the model using Wandb - first results were OK, but better models could be found online
4. To test the model for both images and videos we designed a simple FastAPI backend and a vuejs frontend as a playground
5. I realized that the hardest part was to get a good OCR model. I tried a lot of models and none of them were good enough - but could handle some simple license plates
6. used easyocr, TrOCR finetuned, TrOCR large and tesseract
7. explored ways to preprocess image to improve results (blackhat, deskew, denoise, etc..) gettin better results but still poor in hard datasets


# Project architecture

Basically:


- python fastapi backend loading models and serving it (streaming video img by img - nothing fancy like webRTC, HLS, etc...)
- vuejs frontend to test the model
- s3 localstack service
- machine learning training and testing folders supporting both yolo model training and ocr model testing


More info in folder `readmes`
- [backend](backend/README.md)
- [frontend](frontend/README.md)
- s3 localstack service
- [machine learning yolo models](machine-learning-yolo/README.md)
- [machine learning ocr models](machine-learning-ocr/README.md)

# How to run the project ideally

Docker compose is the easiest way to run the project.

```bash
docker compose up --build
```

this should start the backend, frontend, localstack and get you going

