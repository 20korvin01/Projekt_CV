import cv2
import numpy as np
import matplotlib.pyplot as plt

# load original image
img_original = cv2.imread('original_DSC_0565.png')
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # convert to RGB

# compute binary mask
img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY) # convert to grayscale
_, img_binary = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)

# apply closing operation
kernel = np.ones((9, 9), np.uint8)
closing_mask = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

# compute connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_mask, connectivity=8)
all_labels = labels
# delete connected components with area smaller than 5
area = stats[:, 4] # get area column
posbl_drops = np.array(np.where(area > 5))[0]    # values greater than 5
no_drops_idx = np.array(np.where(area <= 5))[0]  # values smaller than or equal to 5

# delete components with area smaller than 5 in labels.txt
deleted = 0
for i in range(1, num_labels):
    if i in no_drops_idx:
        #? delete component in labels.txt
        labels[labels == i] = 0
        deleted += 1
    #? update labels.txt
    labels[labels == i] = i - deleted
    
num_labels = np.max(labels)
stats = stats[posbl_drops]
centroids = centroids[posbl_drops]


# List to store circularity values
circularity_values = []

# Iterate over each connected component (skipping the background)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y, width, height = stats[i, :4]
    
    # Create a mask for the current component
    component_mask = (labels == i).astype(np.uint8)
    
    # Find the perimeter using cv2.findContours
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularity_values.append(circularity)
        else:
            circularity_values.append(0)  # Handle degenerate cases
    else:
        circularity_values.append(0)
        

