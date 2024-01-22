import cv2
import numpy as np
import matplotlib.pyplot as plt

def depth_based_corners(image_path, depth_map_path):
    # Read the RGB image and depth map
    rgb_image = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Convert depth map to float for better precision
    depth_map = depth_map.astype(float)

    # Threshold the depth map to create binary mask
    _, binary_depth_map = cv2.threshold(depth_map, 220, 255, cv2.THRESH_BINARY)

    # Find connected components in the binary depth map
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_depth_map.astype(np.uint8))

    # Draw rectangles around each segmented object and calculate corners
    image_with_corners = rgb_image.copy()
    for stat in stats[1:]:
        x, y, w, h, _ = stat
        print(len(stat))

        # Skip small components (adjust the threshold as needed)
        if w * h < 100:
            continue

        # Draw the rectangle on the image
        cv2.rectangle(image_with_corners, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Calculate maxCorners based on the number of rectangles (always in multiples of 4)
        # max_corners = int(np.ceil(4 * len(stats) / 4) * 4)
        # print("Max Corners:", max_corners)

        # Extract region of interest for corners calculation
        roi = rgb_image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=12, qualityLevel=0.01, minDistance=10)

        corners = np.intp(corners)

        # Draw corners on the image
        for corner in corners:
            x_corner, y_corner = corner.ravel()
            x_corner += x  # Adjust coordinates to match the original image
            y_corner += y
            cv2.circle(image_with_corners, (x_corner, y_corner), 3, (255, 255, 255), -1)

        for h, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.putText(image_with_corners, text=str(h + 1), org=(int(x), int(y)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                            thickness=2, lineType=cv2.LINE_AA)
        else:
            print("Not enough corners found.")




    # Visualize the depth map with connected components, rectangles, and corners
    plt.figure(figsize=(24, 12))

    # plt.subplot(131), plt.imshow(rgb_image[...,::-1]), plt.title('Original Image')
    # plt.subplot(132), plt.imshow(binary_depth_map, cmap='gray'), plt.title('Binary Depth Map with Connected Components')
    plt.subplot, plt.imshow(image_with_corners[...,::-1]), plt.title('Image with Rectangles and Corners')

    plt.show()

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' and 'your_depth_map.png' with the actual paths to your RGB image and depth map
    image_path = '/home/geetham/Downloads/7_rgb.jpg'
    depth_map_path = '/home/geetham/Downloads/7_depth.jpg'
    depth_based_corners(image_path, depth_map_path)
