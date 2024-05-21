import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load csv file
df = pd.read_csv('CatApproach\labels_approach.csv')
print("csv file loaded. Number of rows:", len(df))

# add header row: img, approach
df.columns = ['img', 'approach']

# check first 3 rows
print(df.head(3))

# count category numbers for approach = y and approach = n
print("Nr of images with cat approaching", len(df[df['approach'] == 'y']))
print("Nr of images without a cat approaching", len(df[df['approach'] == 'n']))
print("Nr of images to exclude", len(df[df['approach'] == 'x']))
print("Nr of images with prey", len(df[df['approach'] == 'p']))

# Exclude rows with 'x' in the 'approach' column
df = df[df['approach'] != 'x']

# Rename 'p' to 'y' in the 'approach' column
df['approach'] = df['approach'].apply(lambda x: 'y' if x == 'p' else x)

# count category numbers for approach = y and approach = n
print("Nr of images with cat approaching", len(df[df['approach'] == 'y']))
print("Nr of images without a cat approaching", len(df[df['approach'] == 'n']))
print("Nr of images to exclude", len(df[df['approach'] == 'x']))
print("Nr of images with prey", len(df[df['approach'] == 'p']))


# Define the path to the "cats_training" folder
image_folder = os.path.join(os.getcwd(), 'CatApproach\cats_approach_training_2024_04_30')
print("Image folder:", image_folder)

# Preprocess images
def preprocess_image(file_name):
    img_path = os.path.join(image_folder, file_name)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read the image file {img_path}. Check file path and integrity.")
    img = cv2.resize(img, (64, 64))  # Resize to uniform size
    img = img / 255.0  # Normalize pixel values
    return img

# Update 'img' column with full paths
df['img'] = df['img'].apply(preprocess_image)
print("Image preprocessing complete.")

# Before splitting data
print("Preparing labels...")
df['approach'] = df['approach'].apply(lambda x: 1 if x == 'y' else 0)

# Split data into training and temporary data (combining validation and test)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

# Split the temporary data into validation and test data
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("Data split into training, validation, and test sets.")
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

"""
# Add dropout layers to the model
# adding dropout layers could improve performance and reduce overfitting

from tensorflow.keras.layers import Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

"""

# Specify a lower learning rate
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print("Model created and compiled.")

# Convert dataframes to numpy arrays for training and validation sets
X_train = np.array(list(train_df['img']))
y_train = np.array(list(train_df['approach']))
X_val = np.array(list(val_df['img']))
y_val = np.array(list(val_df['approach']))

# Train the model
print("Starting model training...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
print("Model training complete. Close image to continue...")

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Model evaluation on test set
# Convert test dataframe to numpy arrays
X_test = np.array(list(test_df['img']))
y_test = np.array(list(test_df['approach']))

# Evaluate the model on the test set
print("Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test set
predictions = model.predict(X_test)
predictions = np.round(predictions).astype(int).flatten()  # Convert probabilities to class labels

# Generate a classification report
report = classification_report(y_test, predictions, target_names=['No Cat approach', 'Cat approach'])
print(report)

# interpreting the results
# Precision: Out of all the images the model predicted to be positive, what fraction were actually positive? High precision indicates a low false positive rate.
# Recall (Sensitivity): Out of all the positive images, what fraction were correctly predicted as positive? High recall indicates a low false negative rate.
# F1 score: Weighted average of precision and recall

# Save the model
model.save('cat_approach_detector_model.h5')  # Saves as an HDF5 file
