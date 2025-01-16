from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def get_dominant_color(image, box, seg_mask=None):
    # Extract the region inside the bounding box
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    
    # Apply median blur to reduce noise
    roi = cv2.medianBlur(roi, 5)
    
    # If segmentation mask is provided, apply it to ROI
    if seg_mask is not None:
        mask_roi = seg_mask[y1:y2, x1:x2]
        roi = cv2.bitwise_and(roi, roi, mask=mask_roi)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Get median color (ignoring black pixels if mask is provided)
    if seg_mask is not None:
        hsv_values = hsv[mask_roi > 0]
    else:
        hsv_values = hsv.reshape(-1, 3)
    
    if len(hsv_values) == 0:
        return 'unknown', np.zeros((100, 200, 3), dtype=np.uint8)
    
    median_color = np.median(hsv_values, axis=0).astype(np.uint8)
    
    # Reference colors in HSV
    color_references = {
        'red1': np.array([0, 150, 150]),     # Pure red
        'red2': np.array([180, 150, 150]),   # Pure red (wrapped)
        'orange': np.array([15, 150, 150]),  # Pure orange
        'yellow': np.array([30, 150, 150]),  # Pure yellow
        'green': np.array([60, 150, 150]),   # Pure green
        'blue': np.array([120, 150, 150]),   # Pure blue
        'white': np.array([0, 0, 250]),      # Pure white
        'black': np.array([0, 0, 0]),        # Pure black
        'gray': np.array([0, 0, 128]),       # Middle gray
        'silver': np.array([0, 0, 192])      # Light gray
    }
    
    # Function to calculate color distance in HSV space
    def color_distance(c1, c2):
        # Special handling for hue
        if c1[0] > 170 and c2[0] < 10:  # Red wrap-around case
            hue_diff = min(abs(c1[0] - c2[0]), abs(c1[0] - 180 - c2[0]))
        elif c1[0] < 10 and c2[0] > 170:
            hue_diff = min(abs(c1[0] - c2[0]), abs(c1[0] + 180 - c2[0]))
        else:
            hue_diff = abs(c1[0] - c2[0])
        
        # Weight the components differently
        hue_weight = 1.0 if c1[1] > 30 else 0.1  # Less weight to hue if low saturation
        weights = np.array([hue_weight, 2.0, 1.0])  # Weights for H, S, V
        diff = np.array([hue_diff, abs(c1[1] - c2[1]), abs(c1[2] - c2[2])])
        return np.sum(weights * diff)
    
    # Find the closest color
    min_dist = float('inf')
    dominant_color = 'unknown'
    
    # First check achromatic colors (black, white, gray, silver)
    if median_color[1] < 30:  # Low saturation
        if median_color[2] < 50:
            dominant_color = 'black'
        elif median_color[2] > 200:
            dominant_color = 'white'
        elif median_color[2] > 150:
            dominant_color = 'silver'
        else:
            dominant_color = 'gray'
    else:
        # Check chromatic colors
        for color_name, ref_color in color_references.items():
            if color_name in ['white', 'black', 'gray', 'silver']:
                continue
            
            dist = color_distance(median_color, ref_color)
            if dist < min_dist:
                min_dist = dist
                dominant_color = color_name.replace('1', '').replace('2', '')
    
    # Create visualization
    color_vis = np.zeros((100, 200, 3), dtype=np.uint8)
    # Original ROI on the left
    roi_resized = cv2.resize(roi, (100, 100))
    color_vis[:100, :100] = roi_resized
    
    # Median color on the right
    median_color_bgr = cv2.cvtColor(np.uint8([[median_color]]), cv2.COLOR_HSV2BGR)[0, 0]
    color_vis[:100, 100:] = median_color_bgr
    
    # Add color information
    cv2.putText(
        color_vis,
        f"H:{median_color[0]} S:{median_color[1]} V:{median_color[2]}",
        (105, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )
    
    return dominant_color, color_vis

def init_model():
    # Initialize YOLO11n-seg model
    model = YOLO('yolo11n-seg.pt')  # Using YOLO11n segmentation version
    return model

def process_video(model, video_path, output_path=None):
    # Define traffic-related classes
    traffic_classes = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        11: "stop sign",
        12: "parking meter"
    }
    
    # Open video file or camera stream
    cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else 0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame with segmentation
        results = model(frame, verbose=False)
        
        # Create visualization windows
        color_analysis = np.zeros((frame.shape[0], 200, 3), dtype=np.uint8)
        segmentation_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        color_y_offset = 0
        
        # Process results
        for result in results:
            if result.masks is None:  # Skip if no segmentation masks
                continue
                
            boxes = result.boxes
            masks = result.masks
            
            for box, mask in zip(boxes, masks):
                class_id = int(box.cls.cpu().numpy()[0])
                # Filter for traffic-related classes
                if class_id in traffic_classes:
                    coords = box.xyxy.cpu().numpy()[0]
                    confidence = box.conf.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Get segmentation mask and resize to frame size
                    seg_mask = mask.data.cpu().numpy()[0]
                    seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                    seg_mask = (seg_mask > 0.5).astype(np.uint8)  # Threshold the mask
                    
                    # Draw bounding box with different colors based on class
                    color = (0, 255, 0)  # Default green for vehicles
                    if class_id in [9, 11, 12]:  # Traffic infrastructure
                        color = (0, 0, 255)  # Red for infrastructure
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Apply segmentation mask with color
                    mask_color = np.zeros_like(frame)
                    mask_color[:] = color
                    segmentation_mask = cv2.add(segmentation_mask, 
                                              cv2.bitwise_and(mask_color, mask_color, mask=seg_mask))
                    
                    # Get vehicle color if it's a vehicle
                    vehicle_color = ""
                    color_vis = None
                    if class_id in [1, 2, 3, 5, 7]:  # If it's a vehicle
                        vehicle_color, color_vis = get_dominant_color(frame, coords, seg_mask)
                        vehicle_color = f" ({vehicle_color})"
                        
                        # Add color visualization to color analysis window
                        if color_vis is not None and color_y_offset + 100 <= color_analysis.shape[0]:
                            color_analysis[color_y_offset:color_y_offset + 100, :200] = color_vis
                            cv2.putText(
                                color_analysis,
                                f"{traffic_classes[class_id]}{vehicle_color}",
                                (5, color_y_offset + 95),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1
                            )
                            color_y_offset += 110
                    
                    # Add label with class name and color for vehicles
                    label = f"{traffic_classes[class_id]}{vehicle_color}: {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
        
        # Blend segmentation mask with original frame
        frame_with_mask = cv2.addWeighted(frame, 0.7, segmentation_mask, 0.3, 0)
        
        # Write frame if output path is provided
        if writer:
            writer.write(frame_with_mask)
        
        # Display frames
        cv2.imshow('Traffic Detection with YOLO11n-seg', frame_with_mask)
        cv2.imshow('Color Analysis', color_analysis)
        cv2.imshow('Segmentation Mask', segmentation_mask)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize model
    model = init_model()
    
    # Process the sample video
    video_path = str(Path(__file__).parent / "data/obj_counting_sample.mp4")
    output_path = str(Path(__file__).parent / "data/output_detection.mp4")
    process_video(model, video_path, output_path)
