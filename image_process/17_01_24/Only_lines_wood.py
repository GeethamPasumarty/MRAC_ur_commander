import cv2
import numpy as np
from geometry_msgs.msg import Point
import rospy

class WoodenBlockDetector:
    def __init__(self):
        # ROS Publisher for wooden block detection result
        self.block_pub = rospy.Publisher("/wooden_block_detection", Point, queue_size=10)

    def detect_wooden_blocks(self, image_path):
        cv_image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define a color range for brown (you may need to adjust these values)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 255])

        # Create a mask using the color range
        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has four vertices (a rectangle)
            if len(approx) == 4:
                # Draw a bounding box around the wooden block
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Publish the result (you might want to publish more information)
                block_msg = Point()
                block_msg.x = (x + x + w) / 2
                block_msg.y = (y + y + h) / 2
                rospy.loginfo("Wooden Block Detected: {}".format(block_msg))
                self.block_pub.publish(block_msg)

        # Display the image with wooden blocks highlighted
        cv2.imshow("Image with Wooden Blocks", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('wooden_block_detector', anonymous=True)
    image_path = '/home/geetham/Downloads/wood.jpg'  # Replace with the actual path to your image file
    wbd = WoodenBlockDetector()
    wbd.detect_wooden_blocks(image_path)
