import csv
import os
from datetime import datetime

def generate_satellite_paths(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} nahi mili!")
        return

    with open(input_path, mode='r', encoding='utf-8') as infile, \
         open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        # Naye columns add karne ke liye fieldnames update karein
        fieldnames = reader.fieldnames + ['timestamp', 's3_prefix']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for row in reader:
            try:
                # 1. Timestamp banana
                date_str = row['acq_date']
                time_str = row['acq_time'].zfill(4)
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H%M")
                
                # 2. S3 Path logic
                doy = dt.timetuple().tm_yday
                s3_prefix = f"noaa-goes16/ABI-L1b-RadC/{dt.year}/{doy:03d}/{dt.hour:02d}/"
                
                # Row update karein
                row['timestamp'] = dt.strftime("%Y-%m-%d %H:%M")
                row['s3_prefix'] = s3_prefix
                
                writer.writerow(row)
                count += 1
                
                # Sample print for first 5 rows
                if count <= 5:
                    print(f"Matched: {row['timestamp']} -> {s3_prefix}")
                    
            except Exception as e:
                continue

        print(f"\nSUCCESS: {count} records aligned!")

if __name__ == "__main__":
    in_file = "data/processed/merged_canada_fires.csv"
    out_file = "data/processed/aligned_metadata.csv"
    generate_satellite_paths(in_file, out_file)