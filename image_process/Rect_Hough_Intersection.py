import cv2
import numpy as np
from geometry_msgs.msg import Point
import rospy

class RectangleIntersectionDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Point, queue_size=10)

    def detect_intersection(self, image_path):
        cv_image = cv2.imread(image_path)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use HoughLines to detect lines in the image
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None and len(lines) >= 2:
            # Filter and extract two most dominant lines
            lines = lines[:2]

            rectangles = []

            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Draw the line on the image
                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Fit a rectangle to the line
                rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2]]))
                rectangles.append(rect)

            # Calculate intersection point of the centerlines
            intersection_point = self.find_intersection(rectangles[0], rectangles[1])

            if intersection_point is not None:
                # Publish the intersection point
                intersection_msg = Point()
                intersection_msg.x = intersection_point[0]
                intersection_msg.y = intersection_point[1]
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)
            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with lines and rectangles
            cv2.imshow("Image with Lines and Rectangles", cv_image)
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two lines detected. Unable to find intersection.")

    def find_intersection(self, rect1, rect2):
        # Get the center coordinates of the rectangles
        center1 = rect1[0]
        center2 = rect2[0]

        # Calculate the intersection point as the midpoint between the two centers
        intersection_point = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)

        return intersection_point

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    # Specify the path to the image file
    image_path = '/home/geetham/Downloads/2.png'  # Replace with the actual path to your image file

    # Create RectangleIntersectionDetector instance
    rid = RectangleIntersectionDetector()

    # Detect intersection in the specified image
    rid.detect_intersection(image_path)
