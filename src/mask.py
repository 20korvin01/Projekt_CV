import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def primitive_thresholding(image, lower, upper):
    '''Thresholding using a simple lower and upper threshold'''
    _, thresh_mask = cv.threshold(image, lower, upper, cv.THRESH_BINARY)
    return thresh_mask

def opening(image, kernel_size):
    '''Opening operation using cv2.morphologyEx'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return opening

def closing(image, kernel_size):
    '''Closing operation using cv2.morphologyEx'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closing

def connected_components(image):
    '''Connected components using cv2.connectedComponentsWithStats'''
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    return num_labels, labels, stats, centroids


# looping through the images
input_folder = "data/Rain Datasets/flash downsized"
output_folder = "data/Output/flash/closing"

for filename in os.listdir(input_folder):
    # read the image
    img_bgr = cv.imread(os.path.join(input_folder, filename))
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    # image to grayscale
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

    # # image to hsv
    # img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

    # apply primitive thresholding
    primitive_mask = primitive_thresholding(img_gray, 250, 255)
    primitive_mask_inv = cv.bitwise_not(primitive_mask)

    # # show primitive_mask_inv using matplotlib
    # plt.imshow(primitive_mask_inv, cmap='gray')
    # plt.title('Primitive Thresholding Inverted')
    # plt.show() 

    # apply closing operation
    closing_mask = closing(primitive_mask, 9)
    closing_mask_inv = cv.bitwise_not(closing_mask)
    
    # # save closing_mask
    # output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    # cv.imwrite(output_path, closing_mask_inv)
    
    print(f"Saved mask for {filename}")
    
    # calculate connected components
    num_labels, labels, stats, centroids = connected_components(closing_mask)
    np.savetxt(f"data/Output/flash/connected_components/stats/{os.path.splitext(filename)[0]}.txt", stats, fmt='%d')
    np.savetxt(f"data/Output/flash/connected_components/centroids/{os.path.splitext(filename)[0]}.txt", centroids, fmt='%d')
    np.savetxt(f"data/Output/flash/connected_components/labels/{os.path.splitext(filename)[0]}.txt", labels, fmt='%d')
    
    
    # print(num_labels)
    # print(labels)
    # print(stats)
    # print(centroids)
    
    # write stats to file
    # np.savetxt("stats.txt", stats, fmt='%d')
    # np.savetxt("centroids.txt", centroids, fmt='%d')
    # np.savetxt("labels.txt", labels, fmt='%d')  
        
    
    # # Visualize components by coloring them
    # colored_components = np.zeros((*closing_mask.shape, 3), dtype=np.uint8)
    # for i in range(1, num_labels):  # Skip background
    #     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    #     colored_components[labels == i] = color

    # # save colored_components
    # path = os.path.join("Output/flash/colored_components", f"{os.path.splitext(filename)[0]}.png")
    # cv.imwrite(path, colored_components)
    
    # break
    
    # show closing_mask using matplotlib
    # plt.imshow(closing_mask, cmap='gray')
    # plt.title('Closing Operation')
    # plt.show()

    # # cv.imwrite('closing_mask.png', closing_mask)

    # # # display img, primitive_mask, opening_mask side by side
    # # plt.figure(figsize=(15, 5))
    # # plt.subplot(131)
    # # plt.imshow(img, cmap='gray')
    # # plt.title('Original Image grayscaled')
    # # plt.subplot(132)
    # # plt.imshow(primitive_mask, cmap='gray')
    # # plt.title('Primitive Thresholding')
    # # plt.subplot(133)
    # # plt.imshow(opening_mask, cmap='gray')
    # # plt.title('Opening Operation')
    # # plt.show()