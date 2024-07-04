import cv2
import numpy as np


def most_frequent_color(image, k=1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Convert to float type for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count the frequency of each cluster center
    unique, counts = np.unique(labels, return_counts=True)
    
    # Find the most frequent cluster center
    most_frequent_center = centers[unique[np.argmax(counts)]]
    
    # Convert back to integer
    most_frequent_center = most_frequent_center.astype(int)
    
    return tuple(most_frequent_center)


def most_frequent_color_masked(image, mask, k=1):
    # Reshape the image and mask to be a list of pixels
    pixels = image.reshape(-1, 3)
    mask_pixels = mask.reshape(-1)
    
    # Filter out the pixels where the mask is zero
    pixels = pixels[mask_pixels != 0]
    
    # Convert to float type for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count the frequency of each cluster center
    unique, counts = np.unique(labels, return_counts=True)
    
    # Find the most frequent cluster center
    most_frequent_center = centers[unique[np.argmax(counts)]]
    
    # Convert back to integer
    most_frequent_center = most_frequent_center.astype(int)
    
    return tuple(most_frequent_center)


def getFeatures(image, binary_mask):
    features = {}

    try:
        col_r, col_g, col_b = most_frequent_color_masked(image, binary_mask, 1)
    except:
        return None
    
    features["blue"] = col_b / 255
    features["green"] = col_g / 255
    features["red"] = col_r / 255

        
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        real_area = cv2.contourArea(largest_contour)
        
        _, radius = cv2.minEnclosingCircle(largest_contour)
        radius = int(radius)
        circle_area = np.pi * radius ** 2

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        rect_area = cv2.contourArea(box)
        
        
        circle_area_ratio = real_area / circle_area
        rect_area_ratio = real_area / rect_area
        
        total_area_ratio = real_area / (image.shape[0] * image.shape[1])
        
        features["circle_area_ratio"] = circle_area_ratio
        features["rect_area_ratio"] = rect_area_ratio
        features["total_area_ratio"] = total_area_ratio
        
        
    else:
        return None
    
    return features
