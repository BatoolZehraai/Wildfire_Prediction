import xarray as xr
import numpy as np
import os

def inspect_nc_data(file_path):
    print(f"--- Inspecting Raw Data: {os.path.basename(file_path)} ---")
    try:
        # Open dataset
        ds = xr.open_dataset(file_path, engine='netcdf4')
        
        print("\n[1] Dimensions:")
        print(ds.dims)
        
        # Get the main data variable (usually 'Rad')
        data_var = list(ds.data_vars)[0]
        data_array = ds[data_var]
        
        # Convert to numpy and filter out NaNs
        raw_values = data_array.values
        valid_mask = ~np.isnan(raw_values)
        valid_data = raw_values[valid_mask]
        
        if valid_data.size > 0:
            print(f"\n[2] Cleaned Data Stats ({data_var}):")
            print(f" -> Max Value: {valid_data.max():.4f}")
            print(f" -> Min Value: {valid_data.min():.4f}")
            print(f" -> Mean Value: {valid_data.mean():.4f}")
            print(f" -> Total Valid Pixels: {valid_data.size}")
            print("\n✅ Success: We have REAL numeric data!")
        else:
            print("\n⚠️ WARNING: This file exists but contains ONLY NaN values.")
            print("Action: NASA server might still be indexing this specific file.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ensure this path matches your downloaded file exactly
    SAMPLE_FILE = "data/raw/satellite_images/OR_ABI-L1b-RadC-M6C01_G16_s20250010001173_e20250010003546_c20250010004006.nc"
    
    if os.path.exists(SAMPLE_FILE):
        inspect_nc_data(SAMPLE_FILE)
    else:
        print(f"File not found at: {SAMPLE_FILE}")