import xarray as xr
import numpy as np
from utils.geo_utils import latlon_to_fixed_grid

ds = xr.open_dataset('data/raw/satellite_images/GOES16_FullDisk_May1_H20.nc')
# Test coordinate (Mid-US)
lat, lon = 39.0, -95.0 
tx, ty = latlon_to_fixed_grid(lat, lon, ds)
ix = np.argmin(np.abs(ds.x.values - tx))
iy = np.argmin(np.abs(ds.y.values - ty))

print(f"Pixel Index for US Center: x={ix}, y={iy}")
print(f"Value at this pixel: {ds.Rad.values[iy, ix]}")