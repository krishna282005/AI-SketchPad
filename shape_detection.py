import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7,
    static_image_mode=False,  # Keep tracking the same hand smoothly
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# Create GUI window
root = tk.Tk()
root.title("AI Sketch Pad")
root.geometry("1000x600")

# Create Canvas
canvas = tk.Canvas(root, bg="white", width=800, height=500)
canvas.pack(side=tk.LEFT, padx=10, pady=10)

# Create Video Feed Label
video_label = tk.Label(root)
video_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Capture webcam
cap = cv2.VideoCapture(0)
last_x, last_y = None, None  # Store last drawing position
drawing = []  # Stores drawn points for shape recognition
drawing_mode = False  # True when pinching
pointer = None  # Pointer to track finger position

def clear_canvas():
    """Clear the entire canvas."""
    canvas.delete("all")

def detect_shape():
    """Detects if the user has drawn a circle or square and replaces it."""
    if len(drawing) < 10:  # Not enough points to detect shape
        return

    points = np.array(drawing)

    # Fit a bounding box around the shape
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width, height = x_max - x_min, y_max - y_min

    # Check if it's a square (Aspect ratio close to 1)
    if 0.9 < (width / height) < 1.1:
        canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="blue", width=3)
    else:  # Otherwise, assume it's a circle
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        radius = max(width, height) // 2
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="red", width=3)

    drawing.clear()  # Reset drawing

# Function to update the hand tracking, drawing, and video feed
def update_frame():
    global last_x, last_y, drawing, drawing_mode, pointer

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Detect hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Convert webcam coordinates to canvas size
            canvas_x = np.interp(x, [0, w], [0, 800])
            canvas_y = np.interp(y, [0, h], [0, 500])

            # Get thumb tip (landmark 4) for pinch detection
            thumb = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)

            # Adaptive pinch threshold based on hand size
            hand_size = np.linalg.norm(
                np.array([hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h]) - 
                np.array([hand_landmarks.landmark[5].x * w, hand_landmarks.landmark[5].y * h])
            )
            pinch_threshold = hand_size * 0.3  # Adjust based on hand size
            pinch_distance = np.linalg.norm(np.array([x, y]) - np.array([thumb_x, thumb_y]))

            # Enable drawing if pinch is detected
            drawing_mode = pinch_distance < pinch_threshold

            # Update the red pointer
            if pointer:
                canvas.delete(pointer)
            pointer = canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5, fill="red")

            if drawing_mode:
                if last_x is not None and last_y is not None:
                    canvas.create_line(last_x, last_y, canvas_x, canvas_y, fill="black", width=3)
                    drawing.append((canvas_x, canvas_y))  # Store points for shape recognition

                last_x, last_y = canvas_x, canvas_y  # Update last position

            else:
                detect_shape()
                last_x, last_y = None, None  # Reset

    # Convert OpenCV frame to Tkinter format for video feed
    img = cv2.resize(rgb_frame, (300, 200))  # Resize for better display
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)  # Update frame every 10ms

# Create a Clear button instead of using a gesture
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas, font=("Arial", 14))
clear_button.pack(side=tk.BOTTOM, pady=10)

# Start hand tracking with video feed
update_frame()

# Run GUI loop
root.mainloop()

# Release resources when the window is closed
cap.release()
cv2.destroyAllWindows()
