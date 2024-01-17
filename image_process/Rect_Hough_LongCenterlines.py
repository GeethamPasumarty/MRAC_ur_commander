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
            # Draw longer sides for each rectangle
            for i, rect in enumerate(rectangles):
                box = np.intp(cv2.boxPoints(rect))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)

                # Draw longer sides
                side1, side2 = self.get_longer_sides(rect)
                cv2.line(cv_image, tuple(map(int, side1)), tuple(map(int, side2)), color, 2)

            # Choose only the longer sides for intersection calculation
            longer_sides = [self.get_longer_sides(rect) for rect in rectangles]

            # Calculate intersection point of the longer sides
            intersection_point = self.find_intersection(*longer_sides[0], *longer_sides[1])

            if intersection_point is not None:
                # Publish the intersection point
                intersection_msg = Point()
                intersection_msg.x = intersection_point[0]
                intersection_msg.y = intersection_point[1]
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)

                # Draw a circle or cross at the intersection point
                cv2.circle(cv_image, (int(intersection_point[0]), int(intersection_point[1])), 5, (255, 0, 0), -1)

            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with rectangles, longer sides, and intersection point
            cv2.imshow("Image with Rectangles, Longer Sides, and Intersection", cv_image)
            cv2.imshow("Depth Map", depth_map)  # Visualize the depth map
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two rectangles detected. Unable to find intersection.")

    def find_intersection(self, line1_start, line1_end, line2_start, line2_end):
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

    def get_longer_sides(self, rect):
        box = np.intp(cv2.boxPoints(rect))

        # Get the longer sides
        side1 = (box[0], box[1])
        side2 = (box[2], box[3])

        # Return longer sides
        return side1, side2

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    # Specify the path to the image and depth map files
    image_path = '/home/geetham/Downloads/8_rgb.jpg'  # Replace with the actual path to your image file
    depth_map_path = '/home/geetham/Downloads/8_depth.png'  # Replace with the actual path to your depth map file

    # Create RectangleIntersectionDetector instance
    rid = RectangleIntersectionDetector()

    # Detect intersection in the specified image
    rid.detect_intersection(image_path, depth_map_path)
