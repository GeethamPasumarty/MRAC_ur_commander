from PIL import Image
from collections import Counter

def get_top_rgb_values(image_path, num_values=5, threshold=10):
    # Open the image
    img = Image.open(image_path)

    # Get all pixel values from the image
    pixels = list(img.getdata())

    # Count the occurrences of each RGB value
    rgb_counter = Counter(pixels)

    # Filter out RGB values that occur less than the threshold
    filtered_rgb = {rgb: count for rgb, count in rgb_counter.items() if count >= threshold}

    # Filter out RGB values with alpha channel equal to 255
    rgb_values = [rgb[:3] for rgb in filtered_rgb if rgb[3] == 255]

    # Get the lightest color (maximum values)
    lightest_color = tuple(max(channel_values) for channel_values in zip(*rgb_values))

    # Get the darkest color (minimum values)
    darkest_color = tuple(min(channel_values) for channel_values in zip(*rgb_values))

    return lightest_color, darkest_color

# Example usage
image_path = '/home/geetham/Downloads/Sticks_2.png'
lightest_color, darkest_color = get_top_rgb_values(image_path)

# Print the results
print("Lightest Color:", lightest_color)
print("Darkest Color:", darkest_color)
