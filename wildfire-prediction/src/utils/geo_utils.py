import numpy as np

def latlon_to_fixed_grid(lat, lon, ds):
    """
    Latitude/Longitude ko Satellite Pixel (x, y) mein convert karta hai.
    """
    # Satellite parameters metadata se uthate hain
    req = ds.goes_imager_projection.attrs
    h = req['perspective_point_height']
    lon_0 = req['longitude_of_projection_origin']
    
    # Degrees to Radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lon_0_rad = np.deg2rad(lon_0)
    
    # Geocentric latitude
    f = 1 / req['inverse_flattening']
    e_sq = f * (2 - f)
    lat_g = np.arctan((1 - e_sq) * np.tan(lat_rad))
    
    # Distance from earth center
    r_c = req['semi_major_axis'] / np.sqrt(1 - e_sq * np.sin(lat_g)**2)
    
    # Intermediate variables
    x_cart = r_c * np.cos(lat_g) * np.cos(lon_rad - lon_0_rad)
    y_cart = r_c * np.cos(lat_g) * np.sin(lon_rad - lon_0_rad)
    z_cart = r_c * np.sin(lat_g)
    
    # Fixed Grid coordinates
    s_x = h - x_cart
    s_y = -y_cart
    s_z = z_cart
    
    # Scan angles
    x = np.arctan(s_y / s_x)
    y = np.arcsin(s_z / np.sqrt(s_x**2 + s_y**2 + s_z**2))
    
    return x, y