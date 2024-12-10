import cv2
import numpy as np
import os

def create_binary_mask(image):
    lower_white = np.array([250, 250, 250], dtype=np.uint8) #for flash, between 250 and 254
    upper_white = np.array([255, 255, 255], dtype=np.uint8) #for flash, 255
    
    mask = cv2.inRange(image, lower_white, upper_white)
    
    # Create a binary image with black pixels within the range and white background
    binary_mask = cv2.bitwise_not(mask) # Invert mask to make white background  (binary_mask = mask for black backgound)
    return binary_mask

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Convert BGR to RGB
            
            # Create and save binary mask
            binary_mask = create_binary_mask(image_rgb)
            
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_path, binary_mask)
            print(f"Saved mask for {filename}")

#Input and output folders
input_folder = "Rain Datasets/flash downsized"
output_folder = "Output/flash"

process_images(input_folder, output_folder)
