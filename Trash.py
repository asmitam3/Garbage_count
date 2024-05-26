import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('Garbage.mp4')

# Minimum contour area for each object type
min_contour_area = {
    'plastic_bottle': 2000,
    'plastic_bowl': 3000,
    'paper_glass': 1500
}

# Initialize Subtractor
algo = cv2.createBackgroundSubtractorMOG2()

# Function to detect objects
def detect_objects(frame):
    # Convert frame to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # Apply background subtraction
    img_sub = algo.apply(blur)
    # Dilate the image
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counters for each object type
    object_counts = {
        'plastic_bottle': 0,
        'plastic_bowl': 0,
        'paper_glass': 0
    }

    # Iterate through contours
    for c in contours:
        # Calculate contour area
        contour_area = cv2.contourArea(c)
        # Check if contour area meets minimum criteria for each object type
        for obj_type, min_area in min_contour_area.items():
            if contour_area > min_area:
                # Draw rectangle around detected object
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Increment object count
                object_counts[obj_type] += 1

    # Draw object counts on the frame
    cv2.putText(frame, f"Plastic Bottles: {object_counts['plastic_bottle']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Plastic Bowls: {object_counts['plastic_bowl']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Paper Glasses: {object_counts['paper_glass']}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Print object counts to shell
    print("Object Counts:")
    for obj_type, count in object_counts.items():
        print(f"{obj_type}: {count}")

    # Return frame with detected objects and counts
    return frame

while True:
    ret, frame = cap.read()

    # Check if the video frame is successfully captured
    if not ret:
        break

    # Detect objects in the frame and draw counts
    frame_with_detection = detect_objects(frame)

    # Display frame with object detection
    cv2.imshow('Object Detection', frame_with_detection)

    # Break the loop if 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
