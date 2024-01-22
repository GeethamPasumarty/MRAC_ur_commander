import rembg
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def remove_background(input_path, output_path):
    # Read the input image file
    with open(input_path, "rb") as input_file:
        input_data = input_file.read()

    # Remove the background using rembg
    output_data = rembg.remove(input_data)

    # Save the output image
    with BytesIO(output_data) as output_file:
        image = Image.open(output_file)
        image.save(output_path)

    # Visualize input and output images
    input_image = Image.open(input_path)
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Output Image")
    plt.imshow(image)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    # Replace these paths with your input and output image paths
    input_image_path = "/home/geetham/Downloads/Sticks_3.jpg"
    output_image_path = "/home/geetham/Outputs/Sticks_3_trspt.png"

    # Call the function to remove the background and visualize the images
    remove_background(input_image_path, output_image_path)
