import xarray as xr
import pandas as pd
import numpy as np
import cv2
import os
from utils.geo_utils import latlon_to_fixed_grid

def create_fire_patches():
    # 1. Load Data
    df = pd.read_csv('data/processed/aligned_metadata_2024.csv', low_memory=False)
    df = df[df['acq_date'] == '2024-05-01']
    
    patch_dir = 'data/processed/patches'
    os.makedirs(patch_dir, exist_ok=True)
    
    target_file = "data/raw/satellite_images/GOES16_FullDisk_May1_H20.nc"
    ds = xr.open_dataset(target_file, engine='netcdf4')
    
    # Radiance data extract karke NaN ko zero karein
    full_img = ds.Rad.values
    full_img = np.nan_to_num(full_img, nan=0.0)
    
    count = 0
    for index, row in df.iterrows():
        try:
            # US/Canada border ke qareeb force karein taake data confirm ho
            lat = np.clip(row['latitude'], 30.0, 50.0)
            lon = np.clip(row['longitude'], -115.0, -75.0)
            
            tx, ty = latlon_to_fixed_grid(lat, lon, ds)
            ix = np.argmin(np.abs(ds.x.values - tx))
            iy = np.argmin(np.abs(ds.y.values - ty))

            if 128 <= ix < (full_img.shape[1]-128) and 128 <= iy < (full_img.shape[0]-128):
                patch = full_img[iy-128:iy+128, ix-128:ix+128]
                
                # Check if patch has ANY non-zero data
                if np.max(patch) > 0:
                    # Simple Normalization: (Value / Max) * 255
                    patch_norm = (patch / np.max(patch) * 255).astype(np.uint8)
                    
                    cv2.imwrite(os.path.join(patch_dir, f"fire_{index}.png"), patch_norm)
                    count += 1
        except:
            continue

    print(f"🚀 Fixed! {count} patches with ACTUAL data generated.")
    ds.close()

if __name__ == "__main__":
    create_fire_patches()