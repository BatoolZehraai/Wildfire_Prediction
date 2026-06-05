import numpy as np
import cv2
import os

patch_path = 'data/processed/patches'
files = [f for f in os.listdir(patch_path) if f.endswith('.png')]

if not files:
    print("❌ No patches found in folder!")
else:
    sample = cv2.imread(os.path.join(patch_path, files[0]), 0)
    print(f"File: {files[0]}")
    print(f"Max Brightness: {np.max(sample)}")
    print(f"Min Brightness: {np.min(sample)}")
    print(f"Average: {np.mean(sample)}")
    
    if np.max(sample) == 0:
        print("🚨 Result: Patch is PURE BLACK (All zeros).")
    else:
        print("✅ Result: Patch has data, but contrast is extremely low.")