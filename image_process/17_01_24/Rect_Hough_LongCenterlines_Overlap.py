import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import PIL

def depth_based_corners(image_path, depth_map_path):
    # Read the RGB image and depth map
    rgb_image = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Convert depth map to float for better precision
    depth_map = depth_map.astype(float)

    # Threshold the depth map to create binary mask
    _, binary_depth_map = cv2.threshold(depth_map, 220, 240, cv2.THRESH_BINARY)

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
        cv2.rectangle(binary_depth_map, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Calculate maxCorners based on the number of rectangles (always in multiples of 4)
        # max_corners = int(np.ceil(4 * len(stats) / 4) * 4)
        # print("Max Corners:", max_corners)

        # Extract region of interest for corners calculation
        roi = rgb_image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=12, qualityLevel=0.01, minDistance=10)

        corners = np.intp(corners)
        print(corners)
        b = []

        for t in range (0,12):
            current = corners[t][0].tolist()
            b.append(current)

        # print(b)
        b = np.array(b)
        # print(b)

        # #finding the image size
        # from PIL import Image
        # img = Image.open("/home/geetham/Downloads/Final_Tests/11_rgb.jpg")
        # width, height = img.size

        # print("W =", width)
        # print("H =", height)

        # #Sorting the corners Type_1
        
        # extremes = np.array([[0,0],[width, 0],[0,height],[width,height]])

        # # Calculate distances from each point in 'corners' to each point in 'extremes'
        # distances = np.linalg.norm(b[:, np.newaxis, :] - extremes, axis=2)
        # print(distances)

        # row_sum = []
        # for t in range (0,20):
        #     row_sum.append(np.sum(distances[t, :]))
    
        # print(row_sum)

        # # Get indices that would sort the distances for each point
        # sorted_indices = np.argsort(row_sum, axis=0)
        # print(sorted_indices)

        # # Create a new array of points sorted based on distances from 'extremes'
        # sorted_points = b[sorted_indices]

        # print(sorted_points)
        

        #Sorting the corners Type_2

        # differences = b[:, np.newaxis, :] - b[np.newaxis, :, :]

        # sq_differences = differences ** 2

        # dist_sq = sq_differences.sum(-1)

        # nearest = np.argsort(dist_sq, axis=1)
        # # print(nearest)


        # K = 3
        # nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

        # # print(nearest_partition) 

        # for i in range(b.shape[0]):
        #     for j in nearest_partition[i, :K+1]:
        #         # plot a line from X[i] to X[j]
        #         # use some zip magic to make it happen:
        #         plt.plot(*zip(b[j], b[i]), color='black')


        # Set the number of neighbors you want to find
        K = 4  # Including the point itself

        # Fit the Nearest Neighbors model
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(b)

        # List to store the results
        nearest_neighbors_list = []

        # Iterate through each coordinate in b
        for coordinate in b:
            # Reshape the coordinate array to make it compatible with kneighbors
            coordinate_reshaped = coordinate.reshape(1, -1)

            # Find the distances and indices of the nearest neighbors
            distances, indices = nbrs.kneighbors(coordinate_reshaped)

            # Extract the nearest neighbors (excluding the point itself)
            nearest_neighbors = [tuple(b[i]) for i in indices[0][1:]]

            # Append the result to the list
            nearest_neighbors_list.append(nearest_neighbors)

        # Calculate the sum of distances for each set of coordinates
        sum_distances_list = [sum(np.linalg.norm(np.array(neighbors) - np.array(coordinate)) for neighbors in nearest_neighbors) for coordinate, nearest_neighbors in zip(b, nearest_neighbors_list)]

        # Find the index of the set with the minimum sum of distances
        min_sum_index = np.argmin(sum_distances_list)

        # Get the set with the minimum sum of distances
        min_sum_set = nearest_neighbors_list[min_sum_index]
        min_sum_set1 = nearest_neighbors_list[min_sum_index + 1]

        # Print or use min_sum_set as needed
        print("Set with the minimum sum of distances:")
        for i, (point, neighbors) in enumerate(zip(b, nearest_neighbors_list)):
            print(f"Point {i + 1}: {point}, Nearest Neighbors: {neighbors}, Sum of Distances: {sum_distances_list[i]}")

        # Display the set 1 with the minimum sum of distances
        print("Set with the minimum sum of distances:")
        print(f"Point: {b[min_sum_index]}, Nearest Neighbors: {min_sum_set}, Sum of Distances: {sum_distances_list[min_sum_index]}")


        # Display the set 2 with the minimum sum of distances
        print("Set with the minimum sum of distances:")
        print(f"Point: {b[min_sum_index + 1]}, Nearest Neighbors: {min_sum_set1}, Sum of Distances: {sum_distances_list[min_sum_index + 1]}")


        # Calculation of interssection point:

        min_sum_set_array = np.array(min_sum_set)
        min_sum_set_array1 = np.array(min_sum_set1)


        # Calculate the intersection point using linear algebra

        A = min_sum_set_array[1] - min_sum_set_array[0]
        B = b[min_sum_index] - min_sum_set_array[2]

        # intersection_point = []
        # intersection_point1 = []

        # Check if the two line segments are not parallel
        if np.cross(A, B) != 0:
            t = np.cross(min_sum_set_array[2] - min_sum_set_array[0], B) / np.cross(A, B)
            intersection_point = min_sum_set_array[0] + t * A
            print("Intersection Point:", tuple(intersection_point))
        else:
            print("Line segments are parallel, no intersection point.")


        # Calculate the intersection point using linear algebra

        R = min_sum_set_array1[1] - min_sum_set_array1[0]
        E = b[min_sum_index + 1] - min_sum_set_array1[2]

        # Check if the two line segments are not parallel
        if np.cross(R, E) != 0:
            t = np.cross(min_sum_set_array1[2] - min_sum_set_array1[0], E) / np.cross(R, E)
            intersection_point1 = min_sum_set_array1[0] + t * R
            print("Intersection Point:", tuple(intersection_point1))
        else:
            print("Line segments are parallel, no intersection point.")


        # Draw a circle or cross at the intersection point
        cv2.circle(image_with_corners, (int(intersection_point[0]), int(intersection_point[1])), 5, (0, 255, 0), -1)
        cv2.circle(image_with_corners, (int(intersection_point1[0]), int(intersection_point1[1])), 5, (255, 0, 0), -1)


        # Draw corners on the image
        
        x = 0
        y = 0

        for corner in b:
            x_corner, y_corner = corner
            x_corner += x  # Adjust coordinates to match the original image
            y_corner += y
            cv2.circle(image_with_corners, (x_corner, y_corner), 3, (255, 255, 255), -1)


        for h, corner in enumerate(b):
            x, y = corner
            cv2.putText(image_with_corners, text=str(h + 1), org=(int(x), int(y)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                            thickness=2, lineType=cv2.LINE_AA)


    # Visualize the depth map with connected components, rectangles, and corners
    plt.figure(figsize=(24, 12))
    plt.plot, plt.imshow(rgb_image[...,::-1]), plt.title('Original Image')
    plt.show()
    plt.figure(figsize=(24, 12))
    plt.plot, plt.imshow(binary_depth_map, cmap='gray'), plt.title('Binary Depth Map with Connected Components')
    plt.show()
    plt.figure(figsize=(24, 12))
    plt.plot, plt.imshow(image_with_corners[...,::-1]), plt.title('Image with Rectangles and Corners')
    plt.show()

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' and 'your_depth_map.png' with the actual paths to your RGB image and depth map
    image_path = '/home/geetham/Downloads/Final_Tests/9_rgb.jpg'
    depth_map_path = '/home/geetham/Downloads/Final_Tests/9_depth.jpg'
    depth_based_corners(image_path, depth_map_path)
