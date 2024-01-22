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
            print((area))
            if area > 100 and hierarchy[0, i, 3] == -1:  # Filter based on area and no parent (no overlap)
                rect = cv2.minAreaRect(cnt)
                rectangles.append(rect)



        if len(rectangles) >= 2:
            # Draw centerlines for each rectangle
            for i, rect in enumerate(rectangles):
                box = np.intp(cv2.boxPoints(rect))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)

                for k, point in enumerate(box):
                    if k ==0:
                        colour = (0,0,255)
                    elif k == 1:
                        colour = (0,255,0)
                    elif k ==2:
                        colour = (255,0,0)
                    else:
                        colour = (0,255,255)
                    cv2.circle(cv_image, (int(point[0]), int(point[1])), 5, colour, -1)

                # Draw centerline parallel to the longer sides
                side1_mid, side2_mid = self.get_longer_sides_midpoints(rect)
                cv2.line(cv_image, tuple(map(int, side1_mid)), tuple(map(int, side2_mid)), color, 2)

            # Choose only the longer centerlines for intersection calculation
            longer_centerlines = self.get_longer_centerlines(rectangles)

            # Calculate intersection point of the longer centerlines
            intersection_point = self.find_intersection(*longer_centerlines[0], *longer_centerlines[1])

            if intersection_point is not None:
                # Publish the intersection point
                intersection_msg = Point()
                intersection_msg.x = intersection_point[0]
                intersection_msg.y = intersection_point[1]
                rospy.loginfo("Intersection Point: {}".format(intersection_msg))
                self.ur_pub.publish(intersection_msg)

                # Draw a circle or cross at the intersection point
                cv2.circle(cv_image, (int(intersection_point[0]), int(intersection_point[1])), 5, (235, 52, 219), -1)

            else:
                rospy.logwarn("Unable to find intersection due to parallel lines.")

            # Display the image with rectangles, centerlines, and intersection point
            cv2.imshow("Image with Rectangles, Centerlines, and Intersection", cv_image)
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

    def get_longer_sides_midpoints(self, rect):
        box = np.intp(cv2.boxPoints(rect))

        #Get the midpoints of the longer sides
        if box[1][0]-box[0][0] < box[2][0] - box[1][0] :
            side1_mid = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
            side2_mid = ((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)

        else :
            side1_mid = ((box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2)
            side2_mid = ((box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2)


        return side1_mid, side2_mid

    def get_longer_centerlines(self, rectangles):
        # Calculate lengths of centerlines for each rectangle
        centerlines_lengths = [np.linalg.norm(np.array(self.get_longer_sides_midpoints(rect))) for rect in rectangles]

        # Choose the two centerlines with the maximum lengths
        indices_of_longest_centerlines = np.argsort(centerlines_lengths)[-2:]
        longer_centerlines = [self.get_longer_sides_midpoints(rectangles[i]) for i in indices_of_longest_centerlines]

        return longer_centerlines

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
