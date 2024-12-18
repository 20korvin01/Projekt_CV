"""
delete connected components with area smaller than 5 in labels.txt, stats.txt and centroids.txt
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os




for filename in os.listdir("Output/flash/connected_components/stats")[:]:
    # load stats.txt
    stats = np.loadtxt(f"Output/flash/connected_components/stats/{filename}", dtype=int, delimiter=' ')
    num_labels = stats.shape[0]
    # load labels.txt
    labels = np.loadtxt(f"Output/flash/connected_components/labels/{filename}", dtype=int, delimiter=' ')
    # load centroids.txt
    centroids = np.loadtxt(f"Output/flash/connected_components/centroids/{filename}", dtype=int, delimiter=' ')
        
    # get area column
    area = stats[:, 4]
    # store all area values greater than 5
    posbl_drops = np.array(np.where(area > 5))[0]
    no_drops_idx = np.array(np.where(area <= 5))[0]
    
    # delete components with area smaller than 5 in labels.txt
    deleted = 0
    for i in range(1, num_labels):
        if i in no_drops_idx:
            #? delete component in labels.txt
            labels[labels == i] = 0
            deleted += 1
        #? update labels.txt
        labels[labels == i] = i - deleted
    # delete components with area smaller than 5 in stats.txt and centroids.txt
    stats = stats[posbl_drops]
    centroids = centroids[posbl_drops]
    # update num_labels
    num_labels = np.max(labels)
    # save labels.txt, stats.txt and centroids.txt
    np.savetxt(f"Output/flash/connected_components/greater_5/labels/{filename}", labels, fmt='%i')
    np.savetxt(f"Output/flash/connected_components/greater_5/stats/{filename}", stats, fmt='%i')
    np.savetxt(f"Output/flash/connected_components/greater_5/centroids/{filename}", centroids, fmt='%i')
    
    
    print(f"Saved image for {filename}")
    
    
# # Visualize components only when area smaller 1000
# colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
# for i in range(1, num_labels):
#     color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
#     colored_components[labels == i] = color