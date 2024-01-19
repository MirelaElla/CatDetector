
import cv2 as cv


# open specific image 
#img = cv.imread("Y:/cats/2023_11_24_14_33_53_175130_cat_detected.jpg")

#cv.imshow("Display window", img)
#k = cv.waitKey(0) # Wait for a keystroke in the window


import os
import csv

# Folder containing the cat images
folder_path = 'Y:/cats/'

# Output CSV file
output_file = 'cat_images_classification.csv'

# List to hold the image data
image_data = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Add image name and classification to the list
        image_data.append([filename, 'cat'])

# Write the data to a CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Image Name', 'Classification'])
    # Write image data
    writer.writerows(image_data)

print(f'Data written to {output_file}')