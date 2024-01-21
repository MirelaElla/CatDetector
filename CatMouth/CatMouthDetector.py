import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Load CSV
df = pd.read_csv('labels1.csv')

# Preprocess images
def preprocess_image(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (64, 64))  # Resize to uniform size
    img = img / 255.0  # Normalize pixel values
    return img

df['img'] = df['img'].apply(preprocess_image)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare labels (binary classification: 1 for 'y', 0 for 'n')
train_df['label'] = train_df['label'].apply(lambda x: 1 if x == 'y' else 0)
val_df['label'] = val_df['label'].apply(lambda x: 1 if x == 'y' else 0)

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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert dataframes to numpy arrays
X_train = np.array(list(train_df['img']))
y_train = np.array(list(train_df['label']))
X_val = np.array(list(val_df['img']))
y_val = np.array(list(val_df['label']))

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('cat_mouth_detector.h5')

# Load the model
model = tf.keras.models.load_model('cat_mouth_detector.h5')

# Test model with test data from above

