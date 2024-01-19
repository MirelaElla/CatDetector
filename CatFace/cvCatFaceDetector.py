import os
import cv2 as cv
import csv

# Folder containing the cat images
folder_path = 'Y:/cats/'

# Output CSV file
output_file = 'C:/Users/Mirela/Desktop/CatDetector/CatFace/cat_images_classification.csv'

# Path to the cat face Haar cascade file
cat_cascade_path = 'C:/Users/Mirela/Desktop/CatDetector/CatFace/haarcascade_frontalcatface.xml'

# Load the cat face Haar cascade
cat_cascade = cv.CascadeClassifier(cat_cascade_path)
if cat_cascade.empty():
    print("Error loading Haar cascade. Check the file path.")
    exit(1)
else:
    print("Haar cascade successfully loaded.")

# Function to classify image
def classify_image(image_path):
    img = cv.imread(image_path)
    if img is not None:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(cat_faces) > 0:
            return 'cat_face_detected'
        else:
            return 'no_cat_face_detected'
    else:
        print(f"Unable to read image at {image_path}")
        return 'invalid_image'

# List to hold the image data
image_data = []

# Iterate through all files in the folder
print(f"Scanning directory: {folder_path}")
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Classifying image: {filename}")
        classification = classify_image(file_path)
        image_data.append([filename, classification])
    else:
        print(f"Skipped non-image file: {filename}")

# Write the data to a CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Classification'])
    writer.writerows(image_data)

print(f"Data written to {output_file}")
