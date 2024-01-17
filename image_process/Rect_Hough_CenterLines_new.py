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
        rectangles = self.filter_rectangles(contours, hierarchy)

        if len(rectangles) >= 2:
            # Draw centerlines for each rectangle
            for rect in rectangles:
                self.draw_centerline(cv_image, rect)

            # Calculate intersection point of the centerlines
            intersection_point = self.find_intersection(rectangles[0], rectangles[1])

            if intersection_point is not None:
                # Publish the intersection point
                intersection_msg = self.create_point_message(intersection_point)
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)

                # Draw a circle or cross at the intersection point
                cv2.circle(cv_image, (int(intersection_point[0]), int(intersection_point[1])), 5, (255, 0, 0), -1)

            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with rectangles, centerlines, and intersection point
            cv2.imshow("Image with Rectangles, Centerlines, and Intersection", cv_image)
            cv2.imshow("Depth Map", depth_map)  # Visualize the depth map
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two rectangles detected. Unable to find intersection.")

    def draw_centerline(self, image, rect):
        box = np.intp(cv2.boxPoints(rect))
        box = np.array(box, dtype=np.int32)

        # Choose the first two points of the box as the longer side
        side1 = (box[0], box[1])

        # Calculate the midpoint of the longer side
        mid = ((side1[0][0] + side1[1][0]) // 2, (side1[0][1] + side1[1][1]) // 2)

        # Calculate the slope of the longer side
        slope = (side1[1][1] - side1[0][1]) / (side1[1][0] - side1[0][0])

        # Draw a line parallel to the longer side passing through the midpoint
        length = int(max(np.linalg.norm(np.array(side1[0]) - np.array(side1[1])), 1))  # Corrected line

        p1 = (int(mid[0] - length / 2), int(mid[1] - length / 2 * slope))
        p2 = (int(mid[0] + length / 2), int(mid[1] + length / 2 * slope))

        cv2.line(image, p1, p2, (0, 255, 0), 2)



    def find_intersection(self, rect1, rect2):
        # Get the midpoints of the centerlines for each rectangle
        center1 = self.get_centerline_midpoint(rect1)
        center2 = self.get_centerline_midpoint(rect2)

        # Calculate the intersection point as the average of the centers
        intersection_point = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)

        return intersection_point

    def get_centerline_midpoint(self, rect):
        box = np.intp(cv2.boxPoints(rect))
        box = np.array(box, dtype=np.int32)

        # Choose the first two points of the box as the longer side
        side1 = (box[0], box[1])

        # Calculate the midpoint of the longer side
        mid = ((side1[0][0] + side1[1][0]) // 2, (side1[0][1] + side1[1][1]) // 2)

        return mid

    def filter_rectangles(self, contours, hierarchy):
        # Filter contours based on area and hierarchy
        rectangles = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100 and hierarchy[0, i, 3] == -1:  # Filter based on area and no parent (no overlap)
                rect = cv2.minAreaRect(cnt)
                rectangles.append(rect)

        return rectangles

    def create_point_message(self, point):
        point_msg = Point()
        point_msg.x = point[0]
        point_msg.y = point[1]
        return point_msg

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
