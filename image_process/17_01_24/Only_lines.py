import cv2
import numpy as np
from geometry_msgs.msg import Point
import rospy

class LineDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Point, queue_size=10)

    def detect_lines(self, image_path):
        cv_image = cv2.imread(image_path)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None and len(lines) >= 2:
            lines = lines[:2]

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

            cv2.imshow("Image with Detected Lines", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two lines detected. Unable to visualize.")

if __name__ == '__main__':
    rospy.init_node('line_detector', anonymous=True)
    image_path = '/home/geetham/Downloads/8_rgb.jpg'
    ld = LineDetector()
    ld.detect_lines(image_path)
