import cv2
import numpy as np

class RectangleCornersVisualizer:
    def __init__(self, image_path):
        self.image_path = image_path

    def process_image(self):
        image = cv2.imread(self.image_path)

        if image is None:
            print(f"Error: Unable to read the image at {self.image_path}")
            return

        corners_image = self.visualize_rectangle_corners(image)

        # Display output image with corners visualized
        cv2.imshow('Output Image with Corners', corners_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the output image with corners visualized
        output_path = 'output_corners_image.jpg'
        cv2.imwrite(output_path, corners_image)
        print(f"Output image with corners saved at: {output_path}")

    def visualize_rectangle_corners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners_image = image.copy()

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            for corner in box:
                cv2.circle(corners_image, tuple(corner), 5, (0, 0, 255), -1)

        return corners_image

if __name__ == '__main__':
    image_path = '/home/geetham/Downloads/8_rgb.jpg'  # Replace with the path to your local image
    visualizer = RectangleCornersVisualizer(image_path)
    visualizer.process_image()
