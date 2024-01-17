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

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None and len(lines) >= 2:
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

                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Calculate and publish lines passing through the center point of each rectangle
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

                # Draw the line passing through the center of the rectangle
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.line(cv_image, (center_x, center_y), (center_x, center_y), (0, 255, 0), 2)

                # Publish the center point of the line
                intersection_msg = Point()
                intersection_msg.x = center_x
                intersection_msg.y = center_y
                rospy.loginfo("Line Passing Through Rectangle Center: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)

            cv2.imshow("Image with Lines Through Rectangle Centers", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two lines detected. Unable to find intersection.")

if __name__ == '__main__':
    rospy.init_node('rectangle_intersection_detector', anonymous=True)
    image_path = '/home/geetham/Downloads/4.jpg'  # Replace with the actual path to your image file
    rid = RectangleIntersectionDetector()
    rid.detect_intersection(image_path)
