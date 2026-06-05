import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
import argparse

# S3 configuration for public access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'noaa-goes16'

def download_satellite_data(year, day, hour):
    """
    Downloads GOES-16 satellite data for a specific year, day, and hour.
    Targets May 2025 (Day 121) to match FIRMS wildfire metadata.
    """
    # Construct the directory path on S3
    prefix = f"ABI-L1b-RadC/{year}/{day:03d}/{hour:02d}/"
    save_dir = "data/raw/satellite_images"
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    print(f"--- S3 SCANNING STARTED ---")
    print(f"Targeting: Year {year}, Day {day}, Hour {hour}")
    
    try:
        # List files in the specific hour directory
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        
        if 'Contents' not in response:
            print(f"❌ No files found for path: {prefix}")
            return

        # Get the first available file in that hour slot
        target_key = response['Contents'][0]['Key']
        filename = os.path.basename(target_key)
        local_path = os.path.join(save_dir, filename)
        
        print(f"Downloading: {filename}")
        s3.download_file(BUCKET_NAME, target_key, local_path)
        print(f"✅ SUCCESS! File saved at: {local_path}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GOES-16 Downloader')
    parser.add_argument('--year', type=int, default=2024, help='Year to download') # Added Year
    parser.add_argument('--day', type=int, default=122, help='Day of year')
    parser.add_argument('--hour', type=int, default=18, help='Hour of day')
    
    args = parser.parse_args()
    
    # Pass args.year here instead of hardcoded 2025
    download_satellite_data(args.year, args.day, args.hour)