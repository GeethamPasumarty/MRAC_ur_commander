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
            for rect in rectangles:
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Draw the rectangle and its centerline
                cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 2)

                center_x = np.mean(box[:, 0]).astype(int)
                center_y = np.mean(box[:, 1]).astype(int)
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

                # Draw a circle or cross at the intersection point
                cv2.circle(cv_image, (int(intersection_point[0]), int(intersection_point[1])), 5, (255, 0, 0), -1)
                # Alternatively, you can draw a cross
                cv2.drawMarker(cv_image, (int(intersection_point[0]), int(intersection_point[1])), (255, 0, 0), 
                               markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with rectangles, centerlines, and intersection point
            cv2.imshow("Image with Rectangles, Centerlines, and Intersection", cv_image)
            cv2.waitKey(0)  # Wait until any key is pressed
            cv2.destroyAllWindows()

        else:
            rospy.logwarn("Less than two rectangles detected. Unable to find intersection.")

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
    image_path = '/home/geetham/Downloads/4.jpg'  # Replace with the actual path to your image file

    # Create RectangleIntersectionDetector instance
    rid = RectangleIntersectionDetector()

    # Detect intersection in the specified image
    rid.detect_intersection(image_path)
