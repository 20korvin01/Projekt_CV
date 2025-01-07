import cv2
import numpy as np
import matplotlib.pyplot as plt

# load original image
img_original = cv2.imread('DSC_0564.png')
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
    
cc_all = np.zeros((*all_labels.shape, 3), dtype=np.uint8)
for i in range(1, num_labels):
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cc_all[all_labels == i] = color
    
cc_greater_5 = np.zeros((*labels.shape, 3), dtype=np.uint8)
for i in range(1, np.max(labels)):
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cc_greater_5[labels == i] = color
    
    
# display img_original, img_gray, and thresh_mask using matplotlib
plt.figure(figsize=(10, 5))  # Adjust size for a side-by-side layout
plt.subplot(121)  # Left plot in a 1x2 grid
plt.imshow(cc_all, cmap='gray')
plt.title('All CC')
plt.axis('off')

plt.subplot(122)  # Right plot in a 1x2 grid
plt.imshow(cc_greater_5, cmap='gray')
plt.title('CC Greater 5')
plt.axis('off')

plt.tight_layout()  # Adjust spacing to avoid overlap
plt.show()


    




# ###! PLOTTING ------------------------------------------------------------------------
# # display img_original, img_gray, and thresh_mask using matplotlib
# plt.figure(figsize=(10, 10))  # Adjust size for a 2x2 layout
# plt.subplot(221)  # Top-left
# plt.imshow(img_original)
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(222)  # Top-right
# plt.imshow(img_gray, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')

# plt.subplot(223)  # Bottom-left
# plt.imshow(img_binary, cmap='gray')
# plt.title('Binary Image')
# plt.axis('off')

# plt.subplot(224)  # Bottom-right
# plt.imshow(closing_mask, cmap='gray')
# plt.title('Closing Mask')
# plt.axis('off')

# # plt.tight_layout()  # Automatically adjusts spacing to avoid overlap
# plt.show()
