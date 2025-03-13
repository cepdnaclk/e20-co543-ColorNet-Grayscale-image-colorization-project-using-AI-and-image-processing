import torch
import cv2
import numpy as np
import argparse
from models.colorization_cnn import ZhangColorizationNet
from torchvision import transforms

# Load the trained model
model_path = "colorization_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ZhangColorizationNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def preprocess_L_channel(image_path, target_size=(256, 256)):
    """Preprocesses the L channel for the model"""
    L = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if L is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    L = cv2.resize(L, target_size)
    L = L.astype(np.float32) / 255.0 * 100  # Normalize L channel to [0, 100]
    return L

def transform_L_to_tensor(L_channel):
    """Transform L channel into tensor for the model"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([50.0], [50.0])  # LAB L-channel normalization
    ])
    L_tensor = transform(L_channel)
    return L_tensor.unsqueeze(0)  # Add batch dimension

def colorize_image(image_path, output_path="output.png"):
    L_channel = preprocess_L_channel(image_path)
    L_tensor = transform_L_to_tensor(L_channel).to(device)

    with torch.no_grad():
        ab_channels = model(L_tensor)  # Predict AB channels
        ab_channels = ab_channels.squeeze(0).cpu().numpy().transpose((1, 2, 0))

    # Denormalize AB channels back to [0, 255]
    ab_channels = (ab_channels + 1) * 128

    # Resize predicted AB channels to match the original image size
    ab_channels_resized = cv2.resize(ab_channels, (L_channel.shape[1], L_channel.shape[0]))

    # Create the final LAB image
    img_lab = np.zeros((L_channel.shape[0], L_channel.shape[1], 3), dtype=np.float32)
    img_lab[:, :, 0] = L_channel  # Keep original L channel
    img_lab[:, :, 1:] = ab_channels_resized

    # Convert LAB to RGB
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Save output image
    cv2.imwrite(output_path, (img_rgb * 255).astype(np.uint8))
    print(f"Colorized image saved as {output_path}")

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize a grayscale image")
    parser.add_argument("image_path", type=str, help="Path to the grayscale image")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    args = parser.parse_args()

    colorize_image(args.image_path, args.output)
