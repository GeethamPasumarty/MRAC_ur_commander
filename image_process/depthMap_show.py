import cv2
import matplotlib.pyplot as plt

def visualize_depth_map(depth_map_path):
    # Read the depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Display the depth map
    plt.imshow(depth_map, cmap='viridis')  # You can change the colormap as needed
    plt.colorbar()
    plt.title('Depth Map Visualization')
    plt.show()

if __name__ == "__main__":
    # Specify the path to your depth map
    depth_map_path = '/home/geetham/Downloads/7_depth.jpg'

    # Visualize the depth map
    visualize_depth_map(depth_map_path)
