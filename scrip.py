from yolo import detectCat
import camera 
import cv2
import key
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize first frame
first_frame = None
limit = 0

while True:
    # Capture frame
    frame = camera.capture_frame(cap)

    # Initialize first_frame
    if first_frame is None:
        first_frame = frame
        continue

    # Detect motion
    motion = camera.detect_motion(first_frame, frame)
    if motion:
        limit = limit +1
        print(f"Motion Detected!  {limit}")
        # Display the frame
        if limit >= 20 :
            limit = 0
            print(f"call detector")
            if detectCat(frame) : 
                print("Cat detected!!!!")
    
    # Display the frame
    cv2.imshow('Frame', frame)

    # Update first_frame for the next comparison
    first_frame = frame

    # Wait for a key press and exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()