import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

for filename in os.listdir("Output/flash/connected_components/stats"):
    # load stats.txt
    stats = np.loadtxt(f"Output/flash/connected_components/stats/{filename}", dtype=int, delimiter=' ')
    num_labels = stats.shape[0]
    # load labels.txt
    labels = np.loadtxt(f"Output/flash/connected_components/labels/{filename}", dtype=int, delimiter=' ')
    # get area column
    area = stats[:, 4]
    # store all area values smaller than 1000
    posbl_drops = area[area < 1000]
    # store all area values greater than 1000
    no_drops = area[area >= 1000]
    no_drops_idx = np.array(np.where(area >= 1000))[0]

    # # Visualize components only when area smaller 1000
    colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in range(1, num_labels):
        if i not in no_drops_idx:
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200))
            colored_components[labels == i] = color
        else:
            color = (255, 0, 0)
            colored_components[labels == i] = color
            
    plt.imshow(colored_components)
    plt.title(f"Zusammenhangskomponenten | deleted (red): {no_drops_idx.shape[0]-1} + background")
    plt.savefig(f"Output/flash/connected_components/deleted_components_1000/{os.path.splitext(filename)[0]}.png", dpi=300)
    
    plt.close()
    
    print(f"Saved image for {filename}")