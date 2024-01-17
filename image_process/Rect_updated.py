import cv2
import numpy as np
from geometry_msgs.msg import Point
import rospy

class RectangleIntersectionDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Point, queue_size=10)

    def detect_intersection(self, image_path, depth_map_path, save_output_path):
        cv_image = cv2.imread(image_path)
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        if cv_image is None or depth_map is None:
            rospy.logerr("Unable to read image or depth map.")
            return

        # Use Canny edge detection
        edges = cv2.Canny(cv_image, 50, 150, apertureSize=3)

        # Use findContours to detect contours and hierarchy
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and hierarchy
        rectangles = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100 and hierarchy[0, i, 3] == -1:  # Filter based on area and no parent (no overlap)
                rect = cv2.minAreaRect(cnt)
                rectangles.append(rect)

        if len(rectangles) >= 2:
            # Draw rectangles and visualize midpoints and lines along the longer sides
            for i, rect in enumerate(rectangles):
                box = np.intp(cv2.boxPoints(rect))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv2.drawContours(cv_image, [box], 0, color, 2)

                # Get the midpoints of the longer sides
                side1_mid, side2_mid = self.get_longer_sides_midpoints(rect)

                # Draw the centerline along the longer side
                cv2.line(cv_image, tuple(map(int, side1_mid)), tuple(map(int, side2_mid)), (0, 255, 0), 2)

                # Draw midpoints
                cv2.circle(cv_image, tuple(map(int, side1_mid)), 5, (0, 255, 0), -1)
                cv2.circle(cv_image, tuple(map(int, side2_mid)), 5, (0, 255, 0), -1)

            # Save the output image
            cv2.imwrite(save_output_path, cv_image)

        else:
            rospy.logwarn("Less than two rectangles detected. Unable to find intersection.")

    def get_longer_sides_midpoints(self, rect):
        box = np.intp(cv2.boxPoints(rect))

        # Get the indices of the longer sides
        side_indices = np.argsort([np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]),
                                   np.linalg.norm(box[2] - box[3]), np.linalg.norm(box[3] - box[0])])

        # Get the midpoints of the longer sides
        side1_mid = ((box[side_indices[0]][0] + box[side_indices[1]][0]) / 2,
                     (box[side_indices[0]][1] + box[side_indices[1]][1]) / 2)
        side2_mid = ((box[side_indices[2]][0] + box[side_indices[3]][0]) / 2,
                     (box[side_indices[2]][1] + box[side_indices[3]][1]) / 2)

        return side1_mid, side2_mid

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    # Specify the path to the image and depth map files
    image_path = '/home/geetham/Downloads/8_rgb.jpg'  # Replace with the actual path to your image file
    depth_map_path = '/home/geetham/Downloads/8_depth.png'  # Replace with the actual path to your depth map file
    save_output_path = '/home/geetham/Outputs/try_new.jpg'  # Replace with the desired path to save the output image

    # Create RectangleIntersectionDetector instance
    rid = RectangleIntersectionDetector()

    # Detect intersection in the specified image
    rid.detect_intersection(image_path, depth_map_path, save_output_path)
    # rid.detect_intersection(image_path, depth_map_path)