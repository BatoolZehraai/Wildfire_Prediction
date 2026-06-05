import pandas as pd
import os

def create_2024_metadata():
    """
    Updates the acquisition year from 2025 to 2024 in the metadata 
    to match the downloaded satellite imagery for May 2024.
    """
    input_path = 'data/processed/aligned_metadata.csv'
    output_path = 'data/processed/aligned_metadata_2024.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found!")
        return

    # Load existing 2025 metadata
    df = pd.read_csv(input_path, low_memory=False)
    
    # Change year to 2024 so it matches our May 2024 satellite file
    df['acq_date'] = df['acq_date'].str.replace('2025', '2024')
    
    # Save the updated metadata
    df.to_csv(output_path, index=False)
    print(f"✅ Created: {output_path}")
    print("Now your metadata year matches your satellite file year (2024).")

if __name__ == "__main__":
    create_2024_metadata()