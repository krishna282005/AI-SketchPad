import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("AI Sketch Pad")
root.geometry("800x600")

# Create a canvas for drawing
canvas = tk.Canvas(root, bg="white", width=800, height=500)
canvas.pack(pady=10)

# Function to start drawing
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

# Function to draw on the canvas
def draw(event):
    global last_x, last_y
    canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=3)
    last_x, last_y = event.x, event.y

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Bind mouse events to the canvas
canvas.bind("<Button-1>", start_draw)  # When the mouse is clicked
canvas.bind("<B1-Motion>", draw)  # When the mouse is dragged

# Add a "Clear" button
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Run the Tkinter main loop
root.mainloop()
