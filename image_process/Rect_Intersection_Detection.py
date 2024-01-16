import cv2
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

class RectangleIntersectionDetector:
    def __init__(self):
        self.ur_pub = rospy.Publisher("/ur_joint_position", Point, queue_size=10)

    def detect_intersection(self, image_path):
        cv_image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area (assumes rectangles are the largest objects)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        if len(contours) >= 2:
            rectangles = []

            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Filter approximate polygons with four corners (assumes rectangles)
                if len(approx) == 4:
                    rectangles.append(cv2.minAreaRect(contour))

            for rect in rectangles:
                center, size, angle = rect
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Draw the rectangle and its centerline
                cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 2)

                center_x = int(center[0])
                center_y = int(center[1])
                cv2.line(cv_image, (center_x, center_y), (center_x, center_y), (0, 255, 0), 2)

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

            # Display the image with rectangles and centerlines
            cv2.imshow("Image with Rectangles and Centerlines", cv_image)
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            print("Less than two rectangles detected. Unable to find intersection.")

    def find_intersection(self, rect1, rect2):
        # Get the center coordinates of the rectangles
        center1 = rect1[0]
        center2 = rect2[0]

        # Calculate the intersection point as the midpoint between the two centers
        intersection_point = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)

        return intersection_point

if __name__ == '__main__':
    import rospy

    # Initialize ROS node
    rospy.init_node('rectangle_intersection_detector', anonymous=True)

    # Specify the path to the image file
    image_path = '/home/geetham/Downloads/4.jpg'  # Replace with the actual path to your image file

    # Create RectangleIntersectionDetector instance
    rid = RectangleIntersectionDetector()

    # Detect intersection in the specified image
    rid.detect_intersection(image_path)
