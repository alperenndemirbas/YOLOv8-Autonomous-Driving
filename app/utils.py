import cv2
import numpy as np

# --- CONSTANTS ---
TARGET_SIZE = (640, 640)

# Class mapping for the specific Autonomous Driving dataset
# Used by api.py to map class IDs to human-readable labels
CLASS_MAPPING = {
    0: 'Car', 
    1: 'Truck', 
    2: 'Pedestrian', 
    3: 'Cyclist', 
    4: 'Traffic Light'
}

# Generate a fixed color palette for visualization
# Setting seed ensures the same class always gets the same color
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(100, 3))

def preprocess_image(image, target_size=TARGET_SIZE):
    """
    Prepares the input image for the YOLO model using Letterbox resizing.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        target_size (tuple): Desired output size (width, height).
        
    Returns:
        tuple: (processed_image, meta_info)
        
    NOTE: Normalization (/ 255.0) is intentionally SKIPPED here because
    the loaded model expects pixel values in the [0, 255] range.
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within target_size while maintaining aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding needed to reach target_size (Letterbox)
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add gray border padding
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Store metadata to allow mapping coordinates back to the original image later
    meta = {
        'scale': scale,
        'pad_top': top,
        'pad_left': left,
        'original_dim': (h, w)
    }
    
    # Return image as float32 but keep 0-255 range (No normalization)
    return padded_image.astype(np.float32), meta

def draw_boxes(image, detections):
    """
    Draws bounding boxes and labels on the image based on API results.
    
    Args:
        image (numpy.ndarray): The original image.
        detections (list): List of dictionaries containing 'class', 'confidence', and 'box'.
        
    Returns:
        numpy.ndarray: The image with annotations drawn.
    """
    # Create a copy to avoid modifying the original image
    img = image.copy()

    if not detections:
        return img

    for det in detections:
        label = det['class']
        score = det['confidence']
        box = det['box']

        # Coordinates are already scaled back to original size by the API
        x1, y1, x2, y2 = map(int, box)

        # Select color based on the label hash (consistent colors for same classes)
        color_idx = abs(hash(label)) % len(COLORS)
        color = COLORS[color_idx]

        # Draw Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw Label Background
        text = f"{label} {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # Draw Text
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img