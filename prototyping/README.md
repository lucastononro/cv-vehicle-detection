# Vehicle Detection and Color Analysis Prototype

This prototype implements real-time vehicle detection, segmentation, and color analysis using YOLO11n and OpenCV.

## Features

- Real-time vehicle and traffic object detection
- Instance segmentation using YOLO11n-seg model
- Vehicle color detection using HSV color space analysis
- Multiple visualization windows for analysis

### Supported Objects

- Vehicles:
  - Cars
  - Trucks
  - Buses
  - Motorcycles
  - Bicycles
- Infrastructure:
  - Traffic lights
  - Stop signs
  - Parking meters

### Color Detection

The system can detect the following vehicle colors:
- Red (handles both hue ranges: 0° and 180°)
- Orange
- Yellow
- Green
- Blue
- White
- Black
- Gray
- Silver

Color detection uses a median-based approach with HSV color space analysis:
1. Extracts vehicle region using segmentation mask
2. Calculates median color in HSV space
3. Uses weighted distance metrics for color classification
4. Special handling for achromatic colors (white, black, gray)

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- ultralytics (for YOLO11n)
- opencv-python
- numpy
- torch (CPU version)

## Usage

Run the prototype:
```bash
python prototype_inference.py
```

The script will:
1. Load the YOLO11n-seg model
2. Process the video from `data/obj_counting_sample.mp4`
3. Display three windows:
   - Main window: Detection + segmentation overlay
   - Color Analysis: Shows detected colors
   - Segmentation Mask: Pure segmentation visualization
4. Save the processed video to `data/output_detection.mp4`

### Controls
- Press 'q' to quit the application

## Visualization

The system provides three real-time visualization windows:

1. **Main Detection Window**
   - Shows the original video
   - Bounding boxes around detected objects
   - Segmentation mask overlay
   - Labels with object type, color, and confidence

2. **Color Analysis Window**
   - Shows detected vehicle regions
   - Displays the median color
   - HSV values for debugging
   - Vehicle type and detected color

3. **Segmentation Mask Window**
   - Pure segmentation visualization
   - Different colors for different object types

## Implementation Details

### Color Detection Algorithm

The color detection uses a sophisticated approach:
```python
# HSV Reference Colors
color_references = {
    'red1': [0, 150, 150],    # Pure red
    'red2': [180, 150, 150],  # Pure red (wrapped)
    'orange': [15, 150, 150], # Pure orange
    'yellow': [30, 150, 150], # Pure yellow
    'green': [60, 150, 150],  # Pure green
    'blue': [120, 150, 150],  # Pure blue
    'white': [0, 0, 250],     # Pure white
    'black': [0, 0, 0],       # Pure black
    'gray': [0, 0, 128],      # Middle gray
    'silver': [0, 0, 192]     # Light gray
}
```

Color classification uses:
- Weighted distance metrics for HSV components
- Special handling for red hue wrap-around
- Separate logic for achromatic colors
- Minimum saturation thresholds

### Segmentation

Uses YOLO11n-seg model for precise object segmentation:
- Resizes masks to match frame dimensions
- Applies threshold for binary mask creation
- Blends with original frame for visualization
- Uses masks for more accurate color detection

## Notes

- Color detection accuracy depends on lighting conditions
- Segmentation helps reduce background interference
- Processing speed depends on hardware capabilities
- Multiple objects can be tracked simultaneously 