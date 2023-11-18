import camera 
import cv2
import api
import key
import os
import cv
api_key = key.API_KEY

cap = cv2.VideoCapture(0)

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
        #print("Motion Detected!")
        # Display the frame
        if limit >= 2 :
            #print(f"limit reached  {str(limit)}")
            cv2.imshow('Captured Frame', frame)

            #save frame as frame.jpeg in the current directory
            jpeg_frame = 'frame.jpeg'
             # Check if the file exists and delete it
            if os.path.exists(jpeg_frame):
                os.remove(jpeg_frame)
                
            cv2.imwrite(jpeg_frame, frame) 

           # call openAi 
           # message = api.sendPrompt(
           #      prompt= "Is there a cat in the picture ? say TRUE OR FALSE" , 
           #      file=jpeg_frame , 
           #      openai_api_key=api_key )

            # call COCO Model 
            message = cv.doDeepNeuralNetworkMagic()
            
           # print(f"Antwort war :  {message}" )

            if message :
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!CAT DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                limit = 0   
            else : 
                limit = 0       

        ## call to a AI 
    
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