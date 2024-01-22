import cv2
import numpy as np
import matplotlib.pyplot as plt

def depth_based_corners(rgb_image_path, depth_map_path):
    # Read the RGB image and depth map
    rgb_image = cv2.imread(rgb_image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Convert depth map to float for better precision
    depth_map = depth_map.astype(float)

    # Threshold the depth map to segment objects (adjust the threshold as needed)
    _, binary_depth_map = cv2.threshold(depth_map, 150, 230, cv2.THRESH_BINARY)

    # Find contours in the binary depth map
    contours, _ = cv2.findContours(binary_depth_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around each segmented object and calculate corners
    image_with_corners = rgb_image.copy()
    for contour in contours:
        # Fit a bounding rectangle around the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Draw the rectangle on the image
        cv2.drawContours(image_with_corners, [box], 0, (0, 255, 0), 2)

        # Calculate maxCorners based on the number of rectangles
        max_corners = int(np.ceil(4 * len(contours) / 4) * 4)
        print("Max Corners:", max_corners)

        # Convert the box points to corners
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
        corners = np.intp(corners)

        # Draw corners on the image
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_with_corners, (x, y), 3, 255, -1)

        # Draw corners on the image
        for corner in corners:
            x_corner, y_corner = corner.ravel()
            x_corner += x  # Adjust coordinates to match the original image
            y_corner += y
            cv2.circle(image_with_corners, (x_corner, y_corner), 3, (255, 0, 0), -1)  # Blue color for circles

    # Visualize the result
    plt.figure(figsize=(12, 6))

    plt.subplot(131), plt.imshow(rgb_image[...,::-1]), plt.title('Original RGB Image')
    plt.subplot(132), plt.imshow(binary_depth_map, cmap='gray'), plt.title('Binary Depth Map with Contours')

    # Draw contours on the binary depth map
    cv2.drawContours(image_with_corners, contours, -1, (0, 0, 255), 2)

    plt.subplot(133), plt.imshow(image_with_corners[...,::-1]), plt.title('Image with Rectangles and Corners')

    plt.show()

if __name__ == "__main__":
    # Replace 'your_rgb_image.jpg' and 'your_depth_map.png' with the actual paths to your RGB image and depth map
    rgb_image_path = '/home/geetham/Downloads/7_rgb.jpg'
    depth_map_path = '/home/geetham/Downloads/7_depth.jpg'
    depth_based_corners(rgb_image_path, depth_map_path)
