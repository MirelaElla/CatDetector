import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Loading COCO class labels
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detectCat(img):
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and (classes[class_id] == "cat" or classes[class_id] == "dog"):
                return True
    return False

