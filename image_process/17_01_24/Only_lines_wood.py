import cv2
import numpy as np
import matplotlib as mp 
from geometry_msgs.msg import Point
import rospy

class WoodenBlockDetector:
    def __init__(self):
        # ROS Publisher for wooden block detection result
        self.block_pub = rospy.Publisher("/wooden_block_detection", Point, queue_size=10)

    def detect_wooden_blocks(self, image_path):
        cv_image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define a color range for brown and add a tolerance
        tolerance = 40
        lower_brown = np.array([161 - tolerance, 139 - tolerance, 107 - tolerance])
        upper_brown = np.array([78 + tolerance, 60 + tolerance, 21 + tolerance])

        # Create a mask using adaptive thresholding
        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, adaptive_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the adaptive thresholded mask
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks_detected = False  # Flag to check if any wooden blocks are detected

        for contour in contours:
            # Filter contours based on area
            area = cv2.contourArea(contour)
            if area >= 200:  # Adjust the area threshold as needed
                # Draw a bounding box around the wooden block
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Publish the result (you might want to publish more information)
                block_msg = Point()
                block_msg.x = (x + x + w) / 2
                block_msg.y = (y + y + h) / 2
                rospy.loginfo("Wooden Block Detected: {}".format(block_msg))
                self.block_pub.publish(block_msg)

                # Print information about the wooden block
                rospy.loginfo("Wooden Block Coordinates: x={}, y={}, w={}, h={}".format(x, y, w, h))

                blocks_detected = True

        if not blocks_detected:
            rospy.loginfo("No wooden blocks detected.")

        # Display the image with wooden blocks highlighted
        # Display the thresholded image
        cv2.imshow("Image with Wooden Blocks", cv_image)
        cv2.imshow("Thresholded Image", adaptive_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('wooden_block_detector', anonymous=True)
    image_path = '/home/geetham/Downloads/Sticks_2.jpg'  # Replace with the actual path to your image file
    wbd = WoodenBlockDetector()
    wbd.detect_wooden_blocks(image_path)
