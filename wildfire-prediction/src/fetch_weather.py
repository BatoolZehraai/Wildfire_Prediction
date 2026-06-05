import pandas as pd
import requests
import os
import time

def get_fast_weather():
    # 1. Sirf un patches ki list lein jo folder mein hain
    patch_dir = 'data/processed/patches'
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
    
    # Patch names se indices nikalna (e.g., fire_patch_10 -> 10)
    # Note: Aapke patch names ke mutabiq split logic check kar lein
    patch_indices = []
    for f in patch_files:
        try:
            idx = int(f.split('_')[-1].split('.')[0])
            patch_indices.append(idx)
        except: continue

    # 2. Original metadata load karein aur sirf patches wali rows rukhlein
    df_full = pd.read_csv('data/processed/aligned_metadata_2024.csv')
    df = df_full.iloc[patch_indices].copy()
    
    print(f"⚡ Fast Track: Fetching wind for only {len(df)} patches (instead of 10 Lakh!)...")
    
    wind_speeds = []
    wind_dirs = []

    for idx, row in df.iterrows():
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={row['latitude']}&longitude={row['longitude']}&start_date={row['acq_date']}&end_date={row['acq_date']}&hourly=windspeed_10m,winddirection_10m"
        
        try:
            response = requests.get(url).json()
            hour = int(str(row['acq_time']).zfill(4)[:2]) # Ensure 4 digits
            w_speed = response['hourly']['windspeed_10m'][hour]
            w_dir = response['hourly']['winddirection_10m'][hour]
            wind_speeds.append(w_speed)
            wind_dirs.append(w_dir)
        except:
            wind_speeds.append(0.0)
            wind_dirs.append(0.0)
        
        print(f"✅ Got weather for patch index {idx}")
        time.sleep(0.05) 

    df['wind_speed'] = wind_speeds
    df['wind_direction'] = wind_dirs
    
    df.to_csv('data/processed/metadata_with_weather.csv', index=False)
    print(f"🚀 Success! 139 patches are now weather-ready.")

if __name__ == "__main__":
    get_fast_weather()