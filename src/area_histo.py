import numpy as np
import matplotlib.pyplot as plt
import os

for filename in os.listdir("Output/flash/connected_components/stats"):
    # load stats.txt
    stats = np.loadtxt(f"Output/flash/connected_components/stats/{filename}", dtype=int, delimiter=' ')
    # get area column
    area = stats[:, 4]
    # store all area values smaller than 1000
    area_1000 = area[area < 1000]

    # create histogram of area
    plt.hist(area_1000, bins=100)
    plt.title("Area Histogram")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    
    # plt.show()
        
    # save histogram
    plt.savefig(f"Output/flash/connected_components/area_histos/{os.path.splitext(filename)[0]}.png", dpi=300)
    
    plt.close()
    
    print(f"Saved histogram for {filename}")
    
