import xarray as xr
import numpy as np
import cv2
import os

def preprocess_satellite_data(file_path):
    print(f"--- Preprocessing: {os.path.basename(file_path)} ---")
    try:
        # Load dataset
        ds = xr.open_dataset(file_path, engine='netcdf4')
        data_var = list(ds.data_vars)[0]
        raw_data = ds[data_var].values

        # 1. Cleaning: Replace NaNs with 0
        raw_data = np.nan_to_num(raw_data)

        # 2. Normalization: Scale data for AI (0 to 1) and Image (0 to 255)
        # Formula: (x - min) / (max - min)
        data_min, data_max = raw_data.min(), raw_data.max()
        normalized = (raw_data - data_min) / (data_max - data_min + 1e-8)
        
        # 3. Save as Image using OpenCV (Bypassing Matplotlib)
        img_8bit = (normalized * 255).astype(np.uint8)
        
        # 4. Save a small version to check
        small_img = cv2.resize(img_8bit, (800, 600))
        cv2.imwrite("data/raw/sample_processed_view.png", small_img)
        
        print("\n✅ Preprocessing Test Success!")
        print(f" -> Matrix Shape: {raw_data.shape}")
        print(f" -> Result saved at: data/raw/sample_processed_view.png")

    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")

if __name__ == "__main__":
    SAMPLE_FILE = "data/raw/satellite_images/OR_ABI-L1b-RadC-M6C01_G16_s20250010001173_e20250010003546_c20250010004006.nc"
    if os.path.exists(SAMPLE_FILE):
        preprocess_satellite_data(SAMPLE_FILE)