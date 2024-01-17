import cv2
import numpy as np
from geometry_msgs.msg import Point
import rospy

class RectangleIntersectionDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Point, queue_size=10)

    def detect_intersection(self, image_path, depth_map_path):
        cv_image = cv2.imread(image_path)
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        if cv_image is None or depth_map is None:
            rospy.logerr("Unable to read image or depth map.")
            return

        edges = cv2.Canny(cv_image, 50, 150, apertureSize=3)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100 and hierarchy[0, i, 3] == -1:
                rect = cv2.minAreaRect(cnt)
                rectangles.append(rect)

        rectangles.sort(key=lambda rect: self.get_depth_level(rect, depth_map))

        if len(rectangles) >= 2:
            corners_list = []
            midpoints_list = []

            for rect in rectangles:
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                side1_mid, side2_mid = self.get_two_shorter_sides_midpoints(box)

                corners_list.append(box)
                midpoints_list.append((side1_mid, side2_mid))

            # Draw rectangles and centerlines
            for i, (rect, corners, midpoints) in enumerate(zip(rectangles, corners_list, midpoints_list)):
                box = np.intp(corners)

                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv2.drawContours(cv_image, [box], 0, color, 2)

                # Draw the centerline
                cv2.line(cv_image, tuple(map(int, midpoints[0])), tuple(map(int, midpoints[1])), (0, 255, 0), 2)

                # Draw a line through the midpoints
                midpoint_line_color = (0, 255, 255)  # Yellow color for the midpoint line
                cv2.line(cv_image, tuple(map(int, midpoints[0])), tuple(map(int, midpoints[1])), midpoint_line_color, 2)

            # Display the image with rectangles, centerlines, and lines through midpoints
            cv2.imshow("Image with Rectangles, Centerlines, and Midpoint Lines", cv_image)
            cv2.imshow("Depth Map", depth_map)  # Visualize the depth map
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two rectangles detected. Unable to find intersection.")

    def get_two_shorter_sides_midpoints(self, box):
        side_indices = np.argsort([np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]),
                                   np.linalg.norm(box[2] - box[3]), np.linalg.norm(box[3] - box[0])])

        side1_mid = ((box[side_indices[0]][0] + box[side_indices[1]][0]) / 2,
                     (box[side_indices[0]][1] + box[side_indices[1]][1]) / 2)
        side2_mid = ((box[side_indices[2]][0] + box[side_indices[3]][0]) / 2,
                     (box[side_indices[2]][1] + box[side_indices[3]][1]) / 2)

        return side1_mid, side2_mid

    def get_depth_level(self, rect, depth_map):
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        depths = [depth_map[corner[1], corner[0]] for corner in box]

        depth_level = np.mean(depths)

        return depth_level

if __name__ == '__main__':
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    image_path = '/home/geetham/Downloads/8_rgb.jpg'
    depth_map_path = '/home/geetham/Downloads/8_depth.png'

    rid = RectangleIntersectionDetector()
    rid.detect_intersection(image_path, depth_map_path)
