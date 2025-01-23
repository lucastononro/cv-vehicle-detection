# Automated License Plate Recognition (ALPR) System

## Disclaimer
**This is a proof-of-concept application and is not intended for production use.**

## Overview
Automated License Plate Recognition (ALPR) system, combining vehicle detection, license plate detection, and Optical Character Recognition (OCR). The project demonstrates the integration of modern computer vision techniques with a production-like architecture.

## Key Features
- Vehicle and license plate detection using YOLOv11
- Multiple OCR approaches for license plate text recognition
- Real-time video processing capabilities
- Interactive web interface for testing and visualization
- Containerized microservices architecture

## System Architecture
The project consists of four main components:

1. **FastAPI Backend**
   - REST API endpoints for detection and OCR processing
   - Real-time video stream processing
   - Model serving and inference
   - Available at `http://localhost:8000`

2. **Vue.js Frontend**
   - Interactive UI for model testing
   - Real-time detection visualization
   - Image and video upload capabilities
   - Available at `http://localhost:5176`

3. **LocalStack S3**
   - Local S3-compatible storage
   - Handles image and video storage
   - Available at `http://localhost:4566`

4. **Machine Learning Components**
   - YOLOv11 custom models for vehicle/plate detection
   - Multiple OCR approaches (EasyOCR, TrOCR, Tesseract, GPT-4o)
   - Image preprocessing pipeline
   - Model training and evaluation tools

## Technical Implementation

### Vehicle Detection
- Custom training YOLOv11 models & pipeline using Roboflow datasets
- Weights & Biases integration (wandb.com) for training performance
- Platform for testing using vue.js as frontend and fastapi as backend
- Automated labeling using heavy-weight api models such as gpt-4o for OCR data collection

### License Plate Recognition
- Multi-stage OCR pipeline
- Supported models:
  - EasyOCR
  - TrOCR (fine-tuned and large variants)
  - Tesseract
  - GPT
- Image preprocessing techniques:
  - Blackhat transformation
  - Deskewingß
  - Denoising
  - Custom preprocessing pipeline

## Getting Started

### Prerequisites
- Docker and Docker Compose
- CUDA-capable GPU (optional, for improved performance)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/cv-vehicle-detection.git

# Start all services
docker compose up --build
```

### Accessing Services
- Backend API: `http://localhost:8000`
  - Swagger UI: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`
- Frontend: `http://localhost:5176`
- LocalStack S3: `http://localhost:4566`

## Detailed Documentation
For more detailed information about specific components:
- [Backend Service](backend/README.md)
- [Frontend Application](frontend/README.md)
- [YOLO Models](machine-learning-yolo/src/README.md)
- [OCR Implementation](machine-learning-ocr/README.md)

## Development Status
This project is actively being developed as a proof of concept. While functional, several areas are being improved:
- OCR accuracy on challenging license plates
- Real-time processing optimization
- Model fine-tuning for specific use cases
