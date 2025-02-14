import cv2
import numpy as np
import csv
import os


def calculate_photo_score(frame, objects):
    """Evaluates the photo-taking score based on scene-wide factors and focus object positioning."""
    h, w, _ = frame.shape

    if not objects:
        return {"Final Score": 1, "Position": 1, "Angle": 1, "Lighting": 1, "Focus": 1,
                "Feedback": ["No subject detected."], "Suggestions": ["Move subject into frame."]}

    # **Scene-Based Scores (Lighting, Angle, Sharpness)**
    # **1️⃣ Angle Score (Horizon Alignment)**
    edges = cv2.Canny(frame, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = [abs(np.degrees(theta)) for rho, theta in lines[:, 0]]
        avg_angle = np.mean(angles)
        angle_score = round(10 - abs(avg_angle - 90) / 9, 2)
    else:
        angle_score = 5

        # **2️⃣ Lighting Score (Scene-Wide Brightness)**
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    lighting_score = round(max(1, min(10, 10 - abs(130 - brightness) / 10)), 2)

    # **3️⃣ Sharpness & Focus Score (Blurriness Check)**
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 50:
        focus_score = 1
    elif variance > 200:
        focus_score = 10
    else:
        focus_score = round((variance - 50) / 15, 2)

        # **Focus Object Position Score**
    focus_object = max(objects, key=lambda obj: obj["confidence"], default=None)
    position_score = 5  # Default neutral
    if focus_object:
        x1, y1, x2, y2 = focus_object["bbox"]
        object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2
        thirds_x, thirds_y = [w // 3, 2 * w // 3], [h // 3, 2 * h // 3]

        dist_x = min(abs(object_x - thirds_x[0]), abs(object_x - thirds_x[1])) / w
        dist_y = min(abs(object_y - thirds_y[0]), abs(object_y - thirds_y[1])) / h
        position_score = round(10 - ((dist_x + dist_y) * 10), 2)

    # **Final Score Calculation**
    final_score = round((position_score * 0.4) + (angle_score * 0.2) +
                        (lighting_score * 0.2) + (focus_score * 0.2), 2)

    # **Generate Feedback & Suggestions**
    feedback = []
    suggestions = []

    if position_score < 5:
        feedback.append("Reposition subject to rule of thirds.")
        if object_x < thirds_x[0]:
            suggestions.append("Move subject to the right.")
        elif object_x > thirds_x[1]:
            suggestions.append("Move subject to the left.")
        if object_y < thirds_y[0]:
            suggestions.append("Move subject lower.")
        elif object_y > thirds_y[1]:
            suggestions.append("Move subject higher.")

    if angle_score < 5:
        feedback.append("Align camera to avoid tilt.")
        suggestions.append("Adjust camera to straighten horizon.")

    if lighting_score < 5:
        feedback.append("Adjust brightness for better exposure.")
        if brightness < 100:
            suggestions.append("Increase lighting or use flash.")
        elif brightness > 180:
            suggestions.append("Reduce exposure to avoid overexposure.")

    if focus_score < 5:
        feedback.append("Hold camera steady to avoid blur.")
        suggestions.append("Use a tripod or stabilize hands.")

    return {
        "Final Score": final_score,
        "Position": position_score,
        "Angle": angle_score,
        "Lighting": lighting_score,
        "Focus": focus_score,
        "Feedback": feedback,
        "Suggestions": suggestions
    }


def save_photo_score(image_path, score_data):
    """Saves the photo-taking score to a CSV file."""
    csv_file = "photo_scores.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Image", "Final Score", "Position", "Angle", "Lighting", "Focus", "Feedback", "Suggestions"])
        writer.writerow([
            image_path, score_data["Final Score"], score_data["Position"], score_data["Angle"],
            score_data["Lighting"], score_data["Focus"], "; ".join(score_data["Feedback"]),
            "; ".join(score_data["Suggestions"])
        ])
