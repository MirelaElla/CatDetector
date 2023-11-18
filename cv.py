import cv2
import numpy as np

def doDeepNeuralNetworkMagic() :

    # Load the model
    net = cv2.dnn.readNetFromCaffe(
        'model/deploy.prototxt',
        'model/mobilenet_iter_73000.caffemodel'
    )

    # Load the image
    image = cv2.imread('frame.jpeg')

    # Prepare the image for the model
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Initialize the boolean flag for cat detection
    cat_detected = False

        # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence of the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.98:  # Adjust the threshold value if necessary
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])
            print(f"Extract the index of the class label from the detections {idx}")

            # Check for cat class index (15 for COCO)
            if idx ==  17 :
                cat_detected = True
                # Optionally, you can print out the bounding box and confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                print(f"Cat detected with confidence: {confidence}, Bounding box: {startX, startY, endX, endY}")
                break  # If we found a cat, no need to check further

    # Return the result
    return cat_detected
