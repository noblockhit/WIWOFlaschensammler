import cv2
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sharedFunctions import most_frequent_color, getFeatures

all_labels = ["braunglas", "gruenglas", "weissglas", "keinglas"]

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

vid = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

model = load_model("model.keras")


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
    features = getFeatures(frame, mask)
    
        
    
    if pressedKey == ord('q'):
        break
    
    
    if features:
        # print(np.array(list(features.values())[:1]).reshape(1, 1))
        # predictions = model.predict(np.array([list(features.values())[0]]).reshape(1, 1), verbose=None)
        predictions = model.predict(np.array(list(features.values())).reshape(1, 6), verbose=None)
        
        print(features)
        print(predictions)
        print(all_labels[np.argmax(predictions[0])])


cv2.destroyAllWindows()
vid.release()