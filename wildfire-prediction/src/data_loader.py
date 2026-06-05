import pandas as pd
import os

def load_canada_fire_data(filename):
    file_path = os.path.join("data", "raw", filename)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"Lead Ahmer: Successfully loaded {len(data)} fire points from Canada.")
        # Filter high confidence fires
        high_conf_data = data[data['confidence'] > 80]
        return high_conf_data
    else:
        print("File not found! Please check data/raw folder.")
        return None

if __name__ == "__main__":
    # Jab file aa jaye toh yahan uska naam likh dena
    # df = load_canada_fire_data('your_file_name.csv')
    pass