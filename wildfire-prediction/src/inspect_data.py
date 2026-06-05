import pandas as pd
import os

# Files ka sahi path (Aapke folder ke mutabiq)
folder_path = "data/raw/"

try:
    # 1. Check karein ke files hain ya nahi
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"--- Files Found: {all_files} ---")

    if not all_files:
        print("Koi CSV file nahi mili! Path check karein.")
    else:
        # 2. Saari files ko merge karein
        df_list = []
        for file in all_files:
            file_full_path = os.path.join(folder_path, file)
            temp_df = pd.read_csv(file_full_path)
            df_list.append(temp_df)
            print(f"Loaded {file}: {len(temp_df)} records")

        df = pd.concat(df_list, ignore_index=True)
        print(f"\n--- Combined Data Summary ---")
        print(f"Total Combined Records: {len(df)}")
        
        # 3. Confidence based filtering
        # Confidence column ko number mein convert karein (jo ghalat characters honge wo NaN ban jayenge)
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        
        # Ab filter karein
        high_conf = df[df['confidence'] > 80]
        print(f"High Confidence Points (>80%): {len(high_conf)}")

        # 4. Location Stats (Canada check)
        print("\n--- Geographic Bounds ---")
        print(f"Lat Range: {df.latitude.min()} to {df.latitude.max()}")
        print(f"Lon Range: {df.longitude.min()} to {df.longitude.max()}")

        # 5. Clean data save karein
        output_path = "data/processed/merged_canada_fires.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Combined data saved to {output_path}")

except Exception as e:
    print(f"Error: {e}")