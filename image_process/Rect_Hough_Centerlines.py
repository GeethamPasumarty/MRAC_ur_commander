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
            # Draw rectangles and centerlines along the longer sides
            for i, rect in enumerate(rectangles):
                box = np.intp(cv2.boxPoints(rect))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv2.drawContours(cv_image, [box], 0, color, 2)

                # Get the midpoints of the longer sides
                side1_mid, side2_mid = self.get_longer_sides_midpoints(rect)

                # Draw the centerline along the longer side
                cv2.line(cv_image, tuple(map(int, side1_mid)), tuple(map(int, side2_mid)), (0, 255, 0), 2)

            # Calculate intersection point of the centerlines
            intersection_point = self.find_intersection(rectangles[0], rectangles[1])

            if intersection_point is not None:
                # Publish the intersection point
                intersection_msg = Point()
                intersection_msg.x = intersection_point[0]
                intersection_msg.y = intersection_point[1]
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)

                # Draw a circle or cross at the intersection point
                cv2.circle(cv_image, (int(intersection_point[0]), int(intersection_point[1])), 5, (255, 0, 0), -1)
                # Alternatively, you can draw a cross
                # cv2.drawMarker(cv_image, (int(intersection_point[0]), int(intersection_point[1])), (255, 0, 0),
                #                markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with rectangles, centerlines, and intersection point
            cv2.imshow("Image with Rectangles, Centerlines, and Intersection", cv_image)
            cv2.imshow("Depth Map", depth_map)  # Visualize the depth map
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

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

    def find_intersection(self, rect1, rect2):
        # Get the midpoints of the longer sides for each rectangle
        side1_mid1, side1_mid2 = self.get_longer_sides_midpoints(rect1)
        side2_mid1, side2_mid2 = self.get_longer_sides_midpoints(rect2)

        # Calculate the intersection point as the intersection of the centerlines
        intersection_point = self.line_intersection(side1_mid1, side1_mid2, side2_mid1, side2_mid2)

        return intersection_point

    def line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        # Calculate the intersection point of two lines
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

        return px, py

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    # Specify the path to the image and depth map files
    image_path = '/home/geetham/Downloads/8_rgb.jpg'  # Replace with the actual path to your image file
    depth_map_path = '/home/geetham/Downloads/8_depth.png'  # Replace with the actual path to your depth map file

    # Create RectangleIntersectionDetector instance
