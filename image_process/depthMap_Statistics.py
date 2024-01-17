import cv2
import matplotlib.pyplot as plt
import numpy as np

depth_map = cv2.imread('/home/geetham/Downloads/8_depth.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(depth_map, cmap='gray')
plt.colorbar()
plt.show()
depth_mean = np.mean(depth_map)
depth_median = np.median(depth_map)
depth_min = np.min(depth_map)
depth_max = np.max(depth_map)

print("Mean Depth:", depth_mean)
print("Median Depth:", depth_median)
print("Min Depth:", depth_min)
print("Max Depth:", depth_max)
