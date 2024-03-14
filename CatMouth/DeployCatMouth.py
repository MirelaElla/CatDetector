import os
import shutil
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('cat_mouth_detector_model.h5')

# Define the source and target directories
src_dir = 'Y:\\cats_testimages'
target_dir = 'Y:\\cats_with_mouths_20'

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate over all images in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load and preprocess the image
        img = Image.open(os.path.join(src_dir, filename))
        img = img.resize((64, 64))  # Resize to match the input shape of the model
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values if your model training used normalization
        img_array = img_array.reshape(1, 64, 64, 3)  # Reshape to match the input shape of the model

        # Predict the presence of a cat mouth
        prediction = model.predict(img_array)
        print(f"Prediction for {filename}: {prediction[0][0]}")

        # If a cat mouth is detected, copy the image to the target directory
        if prediction[0][0] > 0.5:  # Adjust the threshold as needed (lower values copy more images, higher values less)
            shutil.copy(os.path.join(src_dir, filename), target_dir)
            print(f"Cat mouth detected in {filename}. Image copied.")
        else:
            print(f"No cat mouth detected in {filename}.")