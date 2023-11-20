import cv2

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Can't receive frame (stream end?). Exiting ...")
    return frame

def detect_motion(first_frame, second_frame):
    # Convert frames to grayscale
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_second = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    # Blur the frames
    blur_first = cv2.GaussianBlur(gray_first, (21, 21), 0)
    blur_second = cv2.GaussianBlur(gray_second, (21, 21), 0)

    # Compute the difference and apply threshold
    diff_frame = cv2.absdiff(blur_first, blur_second)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes and find contours
    dilated_frame = cv2.dilate(thresh_frame, None, iterations=2)
    contours, _ = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Adjust this threshold for sensitivity
            continue
        motion_detected = True

    return motion_detected