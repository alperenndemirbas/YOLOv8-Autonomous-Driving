import cv2
import numpy as np

# --- CONSTANTS ---
TARGET_SIZE = (640, 640)
# Class mapping for the Self-Driving Car dataset
CLASS_MAPPING = {0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Traffic Light'}
# Visualization colors (BGR format for OpenCV)
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

def preprocess_image(image, target_size=TARGET_SIZE):
    """
    Resizes the image to target_size using 'Letterbox' resizing to maintain aspect ratio.
    Adds padding (gray borders) to fit the square shape without distortion.
    
    Args:
        image (np.array): Original input image (BGR).
        target_size (tuple): Desired output size (width, height).

    Returns:
        padded_image (np.array): Resized and padded image ready for the model.
        meta (dict): Scaling factors and padding values needed for post-processing.
    """
    h, w = image.shape[:2]
    # Calculate scaling factor to fit the image within target_size
    scale = min(target_size[0] / w, target_size[1] / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image preserving aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding needed to reach target_size
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add borders (padding) with neutral gray color (114)
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Store metadata to map bounding boxes back to original coordinates
    meta = {
        'scale': scale,
        'pad_top': top,
        'pad_left': left,
        'original_dim': (h, w)
    }
    
    return padded_image, meta

def draw_boxes(frame, predictions, meta):
    """
    Draws bounding boxes and labels on the original frame using prediction results.
    Uses 'meta' data to reverse the letterbox resizing logic.
    """
    boxes = predictions['boxes'][0]
    classes = predictions['classes'][0]
    confidence = predictions['confidence'][0]
    
    # Retrieve scaling and padding info
    scale = meta['scale']
    pad_left = meta['pad_left']
    pad_top = meta['pad_top']

    valid_indices = np.where(classes != -1)[0]
    
    for i in valid_indices:
        box = boxes[i]
        score = float(confidence[i])
        
        # --- COORDINATE MAPPING (Reverse Letterbox) ---
        # 1. Subtract padding
        # 2. Divide by scale to restore original resolution
        x1 = int((box[0] - pad_left) / scale)
        y1 = int((box[1] - pad_top) / scale)
        x2 = int((box[2] - pad_left) / scale)
        y2 = int((box[3] - pad_top) / scale)

        class_id = int(classes[i])
        label_text = CLASS_MAPPING.get(class_id, 'Unknown')
        label = f"{label_text} {score:.2f}"
        color = COLORS[class_id % len(COLORS)]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background and text for readability
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame