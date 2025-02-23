import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
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
last_x, last_y = None, None  
drawing = []  
drawing_mode = False  
pointer = None  

# Define an eraser mode
eraser_mode = False

def clear_canvas():
    """Clear the entire canvas."""
    canvas.delete("all")
    drawing.clear()

def toggle_eraser():
    """Toggle eraser mode."""
    global eraser_mode
    eraser_mode = not eraser_mode
    eraser_button.config(text="Eraser: ON" if eraser_mode else "Eraser: OFF")


def detect_shape():
    """Detects if the user has drawn a geometric shape and replaces it with a perfect one."""
    if len(drawing) < 10:
        return

    points = np.array(drawing, dtype=np.int32)
    
    # Find the convex hull to avoid self-intersections
    hull = cv2.convexHull(points)

    # Approximate the shape with fewer points
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    num_corners = len(approx)

    if num_corners == 3:
        # Triangle
        points = approx.reshape(-1).tolist()
        canvas.create_polygon(points, outline="green", width=3, fill="")  # <-- Fix

    elif num_corners == 4:
        # Square or Rectangle (Check aspect ratio)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        width, height = x_max - x_min, y_max - y_min

        if 0.9 < (width / height) < 1.1:
            canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="blue", width=3)
        else:
            canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="purple", width=3)

    elif num_corners == 5:
        # Pentagon
        points = approx.reshape(-1).tolist()
        canvas.create_polygon(points, outline="orange", width=3, fill="")  # <-- Fix

    # elif num_corners >= 6:
    #     # Hexagon or more complex shape
    #     points = approx.reshape(-1).tolist()
    #     canvas.create_polygon(points, outline="brown", width=3, fill="")  # <-- Fix

    else:
        # If shape doesn't fit above categories, assume it's a circle
        cx, cy = np.mean(points, axis=0, dtype=int)
        radius = int(np.linalg.norm(points - [cx, cy], axis=1).mean())
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="red", width=3)

    drawing.clear()



# Update hand tracking, drawing, and video feed
def update_frame():
    global last_x, last_y, drawing, drawing_mode, pointer

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)  
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            canvas_x = np.interp(x, [0, w], [0, 800])
            canvas_y = np.interp(y, [0, h], [0, 500])

            thumb = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)

            hand_size = np.linalg.norm(
                np.array([hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h]) - 
                np.array([hand_landmarks.landmark[5].x * w, hand_landmarks.landmark[5].y * h])
            )
            pinch_threshold = hand_size * 0.3  
            pinch_distance = np.linalg.norm(np.array([x, y]) - np.array([thumb_x, thumb_y]))

            drawing_mode = pinch_distance < pinch_threshold

            if pointer:
                canvas.delete(pointer)
            pointer = canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5, fill="red")

            if drawing_mode:
                if last_x is not None and last_y is not None:
                    if eraser_mode:
                        canvas.create_rectangle(canvas_x-10, canvas_y-10, canvas_x+10, canvas_y+10, fill="white", outline="white")
                    else:
                        canvas.create_line(last_x, last_y, canvas_x, canvas_y, fill="black", width=3)
                        drawing.append((canvas_x, canvas_y))

                last_x, last_y = canvas_x, canvas_y  

            else:
                detect_shape()
                last_x, last_y = None, None  

    img = cv2.resize(rgb_frame, (300, 200))  
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)  

# **CLEAR BUTTON**
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas, font=("Arial", 14), bg="red", fg="white")
clear_button.pack(side=tk.BOTTOM, pady=10)

# **ERASER BUTTON**
eraser_button = tk.Button(root, text="Eraser: OFF", command=toggle_eraser, font=("Arial", 14), bg="gray", fg="white")
eraser_button.pack(side=tk.BOTTOM, pady=10)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
