import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sharedFunctions import most_frequent_color
import os
import re


global THRESHOLD
global BLUR
THRESHOLD = 63
BLUR = 41

script_dir = os.path.dirname(os.path.abspath(__file__))

def update_THRESHOLDBar(val):
    global THRESHOLD
    THRESHOLD = val



def update_BLURBar(val):
    global BLUR
    BLUR = val

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 100)
cv2.createTrackbar('Threshold', "Trackbars", THRESHOLD, 100, update_THRESHOLDBar)
cv2.createTrackbar('Blur', "Trackbars", BLUR, 50, update_BLURBar)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

label = input("Please provide a label for the data:")
index = 0


def AnalyseImage(frame):
    global THRESHOLD
    global BLUR

    cv2.imshow("raw", frame)

    mask = np.ones((frame.shape[0],frame.shape[1]))
    mask[:,:] = 255
    common_r, common_g, common_b = most_frequent_color(frame, 1)

    ref_color_space = np.ones((150, 150, 3), dtype=np.uint8)
    ref_color_space[:, :] = [common_b, common_g, common_r]
    
    cv2.imshow("ref color", ref_color_space)

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_color_bgr = np.uint8([[(common_b, common_g, common_r)]])
    target_color_hsv = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Define the HSV range for the target color
    lower_bound = np.array([max(int(target_color_hsv[0]) - THRESHOLD, 0), max(target_color_hsv[1] - THRESHOLD, 0), 50], dtype=np.uint8)
    upper_bound = np.array([min(int(target_color_hsv[0]) + THRESHOLD, 197), min(target_color_hsv[1] + THRESHOLD, 255), 255], dtype=np.uint8)

    # Create a mask for the target color
    mask = cv2.bitwise_not(cv2.inRange(hsv_image, lower_bound, upper_bound))
    mask = cv2.medianBlur(mask, BLUR*2+1)
    cv2.imshow("mask", mask)
    # Use the inverted mask to combine the original image with the black image
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frame, mask

ret, frame = vid.read()
mask = None
while True:
    cv2.imshow("BrarwurstSchnitzelbroetchen",frame)
    pressedKey = cv2.waitKey(50)
    
    ret, frame = vid.read()
    frame, mask = AnalyseImage(frame)
    
    if  pressedKey == ord('q'):
        break
    
        
    elif pressedKey == ord('s'):
        directory_path = os.path.join(script_dir, "TrainingData")
        
        highest_index = -1
        pattern = re.compile(r'.*?(\d+)$')
        for filename in os.listdir(directory_path):
            
            if filename.startswith(label) and filename.endswith('.png'):
                numbers = re.findall(r'\d+', filename)
                numbers = [int(num) for num in numbers]
                if numbers[0] > highest_index:
                    highest_index = numbers[0]
                        
        index = highest_index + 1
        print("pictures:", index+1)
            
        
            
            
        cv2.imwrite(f'TrainingData//{label}_{index}.png',frame)
        cv2.imwrite(f'TrainingData//{label}_{index}.mask.png',mask)
        index +=1



cv2.destroyAllWindows()
vid.release()