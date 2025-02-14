import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")


def detect_objects(frame):
    """Detect objects in an image and return bounding boxes with labels."""
    results = model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            detected_objects.append({"label": label, "confidence": confidence, "bbox": (x1, y1, x2, y2)})

    return detected_objects  # Returns a list of detected objects


def draw_bounding_boxes(frame, objects):
    """Draws bounding boxes around detected objects."""
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = (0, 255, 0)  # Green for detected objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{obj['label']} ({obj['confidence']:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def detect_main_object(frame):
    """Detects the main object and returns the modified frame with bounding box."""
    objects = detect_objects(frame)
    if not objects:
        return frame, None  # Return frame as is if no object is detected

    # Select the main object (highest confidence)
    main_object = max(objects, key=lambda obj: obj["confidence"], default=None)
    frame = draw_bounding_boxes(frame, [main_object]) if main_object else frame
    return frame, main_object
