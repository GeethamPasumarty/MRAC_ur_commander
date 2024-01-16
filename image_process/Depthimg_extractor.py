import cv2
import torch
from torchvision import transforms
from midas import MidasDepthEstimation  # You'll need to install the 'midas' package

def extract_depth_map(rgb_image_path, model_path):
    # Load the MiDaS model
    midas_model = MidasDepthEstimation(model_path)

    # Read the RGB image
    rgb_image = cv2.imread(rgb_image_path)

    # Preprocess the image for the model
    input_image = transforms.ToTensor()(rgb_image)
    input_image = transforms.Resize((384, 384))(input_image)
    input_batch = input_image.unsqueeze(0)

    # Run the model
    with torch.no_grad():
        prediction = midas_model(input_batch)

    # Get the depth map from the prediction
    depth_map = prediction['disp'][0].cpu().numpy()

    return depth_map

if __name__ == "__main__":
    # Specify the path to your RGB image and the MiDaS model
    rgb_image_path = '/home/geetham/Downloads/Sticks_2.jpg'
    model_path = 'dpt_hybrid_384.pt'  # Downloaded the model from https://github.com/intel-isl/MiDaS

    # Extract depth map
    depth_map = extract_depth_map(rgb_image_path, model_path)

    # Normalize the depth map for visualization
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visualization = cv2.applyColorMap(normalized_depth_map.astype(np.uint8), cv2.COLORMAP_JET)

    # Display the depth map
    cv2.imshow("Depth Map", depth_map_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
