import cv2
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sharedFunctions import most_frequent_color, getFeatures

all_labels = {"braunglas": (15, 86, 115),
              "gruenglas": (10, 138, 29),
              "weissglas": (211, 245, 243),
              "keinglas": (0, 0, 255)}

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

model = load_model("model_with_noglass.keras")


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
    ret, frame = vid.read()
    frame, mask = AnalyseImage(frame)
    try:
        features = getFeatures(frame, mask)
    except ZeroDivisionError:
        features = None

    if features:
        predictions = model.predict(np.array(list(features.values())).reshape(1, 6), verbose=None)
        ## get the min enclosing rectangle around the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), list(all_labels.values())[np.argmax(predictions)], 2)
            
            cv2.putText(frame, list(all_labels.keys())[np.argmax(predictions)], (x, y+50), cv2.FONT_HERSHEY_SIMPLEX, .7, list(all_labels.values())[np.argmax(predictions)], 2, cv2.LINE_AA)
    
    cv2.imshow("BrarwurstSchnitzelbroetchen",frame)
    pressedKey = cv2.waitKey(50)
    
    if pressedKey == ord('q'):
        break

cv2.destroyAllWindows()
vid.release()