#!/usr/bin/env python

import cv2
import numpy as np
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64

class LineIntersectionDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Pose, queue_size=10)

    def detect_intersection(self, image_path):
        cv_image = cv2.imread(image_path)
        
        # Apply image processing to detect lines (you may need to adjust parameters)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None and len(lines) >= 2:
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

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.line(cv_image, (center_x, center_y), (center_x, center_y), (0, 0, 255), 2)

            intersection_point = self.find_intersection(lines[0][0], lines[1][0])

            if intersection_point is not None:
                # Continue processing and publishing the intersection point
                intersection_msg = Point()
                intersection_msg.x = intersection_point[0]
                intersection_msg.y = intersection_point[1]
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)
            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")


            # Display the image with centerlines
            cv2.imshow("Image with Centerlines", cv_image)
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            print("Less than two lines detected. Unable to find intersection.")


    def find_intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        B = np.array([rho1, rho2])

        try:
            intersection_point = np.linalg.solve(A, B)
            return intersection_point
        except np.linalg.LinAlgError:
            # Handle the case where the matrix is singular (parallel lines)
            rospy.logwarn("Lines are parallel, no unique intersection point.")
            return None


if __name__ == '__main__':
    import rospy

    # Initialize ROS node
    rospy.init_node('line_intersection_detector', anonymous=True)

    # Specify the path to the image file
    image_path = '/home/geetham/Downloads/Sticks_1.png'  # Replace with the actual path to your image file

    # Create LineIntersectionDetector instance
    lid = LineIntersectionDetector()

    # Detect intersection in the specified image
    lid.detect_intersection(image_path)
