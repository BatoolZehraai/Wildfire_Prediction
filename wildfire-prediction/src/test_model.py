import torch
import cv2
import os
import numpy as np
from train import SimpleUNet

def test_inference():
    # Setup
    device = torch.device("cpu")
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("models/wildfire_diffusion_v1.pth"))
    model.eval()

    # Ek real patch uthayein testing ke liye
    patch_dir = 'data/processed/patches'
    files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
    if not files: return
    
    test_img_path = os.path.join(patch_dir, files[0])
    original = cv2.imread(test_img_path, 0)
    
    # Model input ke liye prepare karein
    img_tensor = torch.from_numpy(original).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(img_tensor)
    
    # Result ko wapas image mein convert karein
    output_np = (output.squeeze().numpy() * 255).astype(np.uint8)
    
    # Original aur Reconstruction ko side-by-side jorein
    comparison = np.hstack((original, output_np))
    cv2.imwrite("comparison_result.png", comparison)
    print("✨ Comparison saved as 'comparison_result.png'.  open nd watch!")

if __name__ == "__main__":
    test_inference()
