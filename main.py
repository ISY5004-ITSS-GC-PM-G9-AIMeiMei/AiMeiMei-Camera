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


def save_photo(raw_frame):
    """Saves the raw camera frame (without the grid) as an image with a timestamp."""
    if not os.path.exists("captured_photos"):
        os.makedirs("captured_photos")  # Create directory if not exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_photos/photo_{timestamp}.jpg"
    cv2.imwrite(filename, raw_frame)
    print(f"Photo saved as: {filename}")


def draw_text(frame, text, position, font_scale=1, thickness=2, color=(255, 255, 255)):
    """Draws outlined text on the frame to ensure visibility."""
    x, y = position
    cv2.putText(frame, text, (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2,
                cv2.LINE_AA)  # Black shadow
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)  # White text


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

capture_message_timer = 0  # Timer to display "Scene Captured" message
capture_message_duration = 30  # Frames (~1 second at 30 FPS)

while True:
    # Capture frame-by-frame
    ret, raw_frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame exactly to match screen resolution
    display_frame = cv2.resize(raw_frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Draw the grid overlay (only for display)
    display_frame = draw_camera_grid(display_frame)

    # Draw instructions at the bottom of the screen
    draw_text(display_frame, "Press 'S' to Capture | Press 'Q' to Quit", (50, screen_height - 50), font_scale=1.2,
              thickness=3)

    # Show "Scene Captured" message if capture was recently taken
    if capture_message_timer > 0:
        draw_text(display_frame, "Scene Captured!", (screen_width // 2 - 100, 100), font_scale=2, thickness=4,
                  color=(0, 255, 0))
        capture_message_timer -= 1

    # Show the frame with grid and instructions
    cv2.imshow("Camera Feed", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('s'):  # Press 's' to take a photo (save without grid)
        save_photo(raw_frame)
        capture_message_timer = capture_message_duration  # Show "Scene Captured" message

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
