import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# Centralized class names and color mapping for consistency across API and Frontend.
CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'pedestrian',
    3: 'cyclist',
    4: 'traffic light'
}

CLASS_COLORS = {
    0: (0, 255, 0),    # Green
    1: (255, 0, 0),    # Blue (OpenCV uses BGR)
    2: (0, 0, 255),    # Red
    3: (255, 0, 255),  # Magenta
    4: (0, 165, 255)   # Orange
}
DEFAULT_COLOR = (255, 255, 255)

def letterbox_image(image, target_size=(640, 640)):
    """
    Resizes the image while maintaining aspect ratio and adding padding (letterboxing).
    Returns the processed image and metadata required for coordinate recovery.

    Args:
        image (PIL.Image): Original input image.
        target_size (tuple): Desired output size (width, height).

    Returns:
        tuple: (new_image, meta_dict)
    """
    iw, ih = image.size
    w, h = target_size
    
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    
    # Create a new image with gray background
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    
    # Paste the resized image at the center
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image.paste(image_resized, (dx, dy))
    
    meta = {
        'scale': scale,
        'pad_left': dx,
        'pad_top': dy
    }
    return new_image, meta

def recover_coordinates(box, meta):
    """
    Maps bounding box coordinates from the resized model input back to the original image.

    Args:
        box (list): [x1, y1, x2, y2] from the model output.
        meta (dict): Metadata containing scale and padding values.

    Returns:
        list: [x1, y1, x2, y2] mapped to original image dimensions.
    """
    x1, y1, x2, y2 = box
    scale = meta['scale']
    pad_left = meta['pad_left']
    pad_top = meta['pad_top']
    
    # Reverse transformation: (Coordinate - Padding) / Scale
    orig_x1 = (x1 - pad_left) / scale
    orig_y1 = (y1 - pad_top) / scale
    orig_x2 = (x2 - pad_left) / scale
    orig_y2 = (y2 - pad_top) / scale
    
    return [orig_x1, orig_y1, orig_x2, orig_y2]

def draw_boxes(image, detections):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR).
        detections (list): List of detection dictionaries containing box, class_id, and score.

    Returns:
        numpy.ndarray: Annotated image.
    """
    img = image.copy()

    for det in detections:
        try:
            # Validate input
            box = det.get('box') 
            if box is None: continue
            
            x1, y1, x2, y2 = map(int, box)
            class_id = int(det.get('class_id', -1))
            score = det.get('score', 0.0)

            # Assign color based on class ID
            color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)

            # Prepare label text
            label_text = CLASS_NAMES.get(class_id, f"Class {class_id}")
            label = f"{label_text} {score:.2f}"

            # 1. Draw Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Calculate text size for background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y1, h)
            
            # 2. Draw Label Background (Filled rectangle)
            cv2.rectangle(img, (x1, y_label - h), (x1 + w, y_label), color, -1)
            
            # 3. Put Text (White color for readability)
            cv2.putText(img, label, (x1, y_label - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
        except Exception as e:
            print(f"Drawing Error: {e}")
            continue

    return img