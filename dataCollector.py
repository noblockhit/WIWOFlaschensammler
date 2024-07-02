import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

global THRESHOLD
global BLUR
THRESHOLD = 255
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

    start1 = time.perf_counter_ns()
    
    blurred = cv2.blur(frame, (BLUR*2+1, BLUR*2+1))
    mask  = np.ones((frame.shape[0],frame.shape[1]))
    mask[:,:] = 255
    
    print(f"Time to blur: {(time.perf_counter_ns()-start1) / 1000000}ms")
    start1 = time.perf_counter_ns()

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
    
    print(f"Time to calculate histograms: {(time.perf_counter_ns()-start1) / 1000000}ms")
    start1 = time.perf_counter_ns()

    
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_color_bgr = np.uint8([[(common_b, common_g, common_r)]])
    target_color_hsv = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Define the HSV range for the target color
    lower_bound = np.array([target_color_hsv[0] - THRESHOLD, 50, 50])
    upper_bound = np.array([target_color_hsv[0] + THRESHOLD, 255, 255])

    # Create a mask for the target color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Invert the mask to get the areas that are not the target color
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverted mask to combine the original image with the black image
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    print(f"Time to process image: {(time.perf_counter_ns()-start1) / 1000000}ms")

    return frame, mask

ret, frame = vid.read()
mask = None
while True:
    cv2.imshow("RichardsFensterXD",frame)
    pressedKey = cv2.waitKey(50)
    
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
