import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the saved model
model = load_model('FaceMask-Detector/face_mask_detector_mobilenetv2.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process and predict mask status for faces in the image
def process_image(file_path):
    try:
        # Load and display the original image
        original_image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

        mask_count = 0
        no_mask_count = 0

        for (x, y, w, h) in faces:
            # Extract face region
            face_img = original_image[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (150, 150))
            face_array = np.expand_dims(face_img, axis=0) / 255.0

            # Predict mask status
            prediction = model.predict(face_array)[0][0]
            label = "NO MASK" if prediction > 0.5 else "MASK"
            color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

            # Count masks and no masks
            if prediction > 0.5:
                no_mask_count += 1
            else:
                mask_count += 1

            # Draw rectangle and label on the image
            cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert image to RGB and display it in the GUI
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(original_image)
        img = ImageTk.PhotoImage(img)

        # Update GUI image panel
        image_panel.configure(image=img)
        image_panel.image = img

        # Display the summary of mask counts
        messagebox.showinfo("Summary", f"Mask Count: {mask_count}\nNo Mask Count: {no_mask_count}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to open file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        process_image(file_path)

# Create the GUI
root = tk.Tk()
root.title("Face Mask Detector")

# Set window size and layout
root.geometry("800x600")

# Title Label
title_label = tk.Label(root, text="Face Mask Detector", font=("Helvetica", 16))
title_label.pack(pady=10)

# Button to select image
select_button = tk.Button(root, text="Select Image", command=select_image, font=("Helvetica", 14))
select_button.pack(pady=20)

# Image panel to display the uploaded image
image_panel = tk.Label(root)
image_panel.pack(pady=10)

# Start the GUI loop
root.mainloop()
