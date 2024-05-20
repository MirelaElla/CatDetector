import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('cat_mouth_detector_model.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open('cat_mouth_detector_model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
