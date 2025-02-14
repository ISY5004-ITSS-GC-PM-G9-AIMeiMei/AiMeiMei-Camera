import cv2
import numpy as np
import screeninfo
import os
from datetime import datetime


def draw_camera_grid(frame):
    """Draws a 3x3 grid on the camera feed."""
    height, width, _ = frame.shape

    # Define grid spacing
    x_step = width // 3
    y_step = height // 3

    # Draw vertical lines
    for i in range(1, 3):
        cv2.line(frame, (i * x_step, 0), (i * x_step, height), (255, 255, 255), 2)

    # Draw horizontal lines
    for i in range(1, 3):
        cv2.line(frame, (0, i * y_step), (width, i * y_step), (255, 255, 255), 2)

    return frame


def set_camera_resolution(cap, width, height):
    """Forces the camera to match the screen's resolution & aspect ratio."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def save_photo(frame):
    """Saves the current frame as an image with a timestamp."""
    if not os.path.exists("captured_photos"):
        os.makedirs("captured_photos")  # Create directory if not exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_photos/photo_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Photo saved as: {filename}")


# Get screen resolution
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height
print(f"Detected Screen Resolution: {screen_width}x{screen_height}")

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution to match screen resolution
set_camera_resolution(cap, screen_width, screen_height)

# Set full-screen mode
cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame exactly to match screen resolution
    frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Draw the grid overlay
    frame = draw_camera_grid(frame)

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('s'):  # Press 's' to take a photo
        save_photo(frame)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
