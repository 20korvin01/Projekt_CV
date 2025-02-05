"""
Line extraction to get the orientation of the patches

idea: Droplets all have a vertical orientation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# load only those parts of the original greyscale image which are currently part of the mask
# to speed up processing. Use zero-padding for background and non-droplet clusters

# image is a greyscale
image = cv2.imread('DSC_0565.png')

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

## Edge direction
# Compute gradients using Sobel filter
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

# Compute gradient magnitude (edge strength)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Compute gradient direction (edge orientation)
gradient_direction = np.arctan2(grad_y, grad_x)  # This gives direction in radians
gradient_direction_deg = np.degrees(gradient_direction)  # Convert to degrees

# np.savetxt(f"Output/{"testdirection"}", gradient_direction_deg,fmt=float)

# only keep the clusters with majority of edge direction in 0 or +-180°
# allow for magrin for angles shots or wind shifting the droplets away from vertical path
# -> maybe shift values by 90° to avoid the jump from 180° to -180
# -- should be easier to implement margins

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
# plt.show()
# Plots for edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap='hot')
plt.title("Gradient Magnitude")
plt.axis('off')

plt.subplot(1, 3, 3)
# plt.imshow(gradient_direction_deg, cmap='hsv')
cax = plt.imshow(gradient_direction_deg, cmap='hsv')
plt.title("Gradient Direction (Degrees)")
plt.axis('off')

# Add colorbar to the gradient direction plot (last subplot)
plt.colorbar(cax, ax=plt.gca(), orientation='vertical', label="Gradient Direction (Degrees)")

plt.tight_layout()
plt.show()
