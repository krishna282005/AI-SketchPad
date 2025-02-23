import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create GUI window
root = tk.Tk()
root.title("AI Sketch Pad")
root.geometry("800x600")

# Create Canvas
canvas = tk.Canvas(root, bg="white", width=800, height=500)
canvas.pack(pady=10)

# Capture webcam
cap = cv2.VideoCapture(0)
last_x, last_y = None, None  # Store last drawing position

# Function to update the webcam feed
def update_frame():
    global last_x, last_y

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
            index_finger = hand_landmarks.landmark[8]  # Only track index finger
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Convert webcam coordinates to canvas size
            canvas_x = np.interp(x, [0, w], [0, 800])
            canvas_y = np.interp(y, [0, h], [0, 500])

            # Draw if finger is detected
            if last_x is not None and last_y is not None:
                canvas.create_line(last_x, last_y, canvas_x, canvas_y, fill="black", width=3)

            last_x, last_y = canvas_x, canvas_y  # Update last position

    # Convert OpenCV frame to Tkinter format
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)  # Update frame every 10ms

# Create a label to display the webcam feed
video_label = tk.Label(root)
video_label.pack()

# Start webcam and GUI loop
update_frame()
root.mainloop()

# Release resources when the window is closed
cap.release()
cv2.destroyAllWindows()
