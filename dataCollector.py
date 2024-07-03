import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

global THRESHOLD
global BLUR
THRESHOLD = 80
BLUR = 35

def update_THRESHOLDBar(val):
    global THRESHOLD     
    THRESHOLD = val

def  update_BLURBar(val):
    global BLUR
    BLUR = val

cv2.namedWindow("Trackbars")

cv2.createTrackbar('Threshold', "Trackbars", THRESHOLD, 100, update_THRESHOLDBar)
cv2.createTrackbar('Blur', "Trackbars", BLUR, 50, update_BLURBar)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

label = input("Please provide a label!!!!!!!! ")
index = 0

def AnalyseImage(frame):
    global THRESHOLD
    global BLUR

    cv2.imshow("raw", frame)

    
    blurred = cv2.blur(frame, (BLUR*2+1, BLUR*2+1))
    mask  = np.ones((frame.shape[0],frame.shape[1]))
    mask[:,:] = 255
    
    height, width, channels = frame.shape

    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

    hist_b = hist_b.flatten()
    hist_g = hist_g.flatten()
    hist_r = hist_r.flatten()
    hists = [hist_b, hist_g, hist_r]

    common_b = np.argmax(hist_b)
    common_g = np.argmax(hist_g)
    common_r = np.argmax(hist_r)
    
    ref_color_space = np.ones((50, 50, 3), dtype=np.uint8)
    ref_color_space[:, :] = [common_b, common_g, common_r]
    
    cv2.imshow("ref color", ref_color_space)

    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_color_bgr = np.uint8([[(common_b, common_g, common_r)]])
    target_color_hsv = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Define the HSV range for the target color
    lower_bound = np.array([max(int(target_color_hsv[0]) - THRESHOLD, 0), max(target_color_hsv[1] - THRESHOLD, 0), 50], dtype=np.uint8)
    upper_bound = np.array([min(int(target_color_hsv[0]) + THRESHOLD, 197), min(target_color_hsv[1] + THRESHOLD, 255), 255], dtype=np.uint8)

    print(lower_bound)
    print(upper_bound)
    print("---")

    # Create a mask for the target color
    mask = cv2.bitwise_not(cv2.inRange(hsv_image, lower_bound, upper_bound))

    cv2.imshow("mask", mask)
    # Use the inverted mask to combine the original image with the black image
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frame, mask

ret, frame = vid.read()
mask = None
while True:
    cv2.imshow("BrarwurstSchnitzelbroetchen",frame)
    pressedKey = cv2.waitKey(500)
    
    ret, frame = vid.read()
    frame, mask = AnalyseImage(frame)
    
    if  pressedKey == ord('q'):
        break
    
        
    elif pressedKey == ord('s'):
        cv2.imwrite(f'TrainingData//{label}_{index}.png',frame)
        cv2.imwrite(f'TrainingData//{label}_{index}.mask.png',mask)
        index +=1



cv2.destroyAllWindows()
vid.release()