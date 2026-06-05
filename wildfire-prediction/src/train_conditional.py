import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os

# 1. Dataset with Proper Normalization
class ConditionalWildfireDataset(Dataset):
    def __init__(self, csv_file, patch_dir):
        self.df = pd.read_csv(csv_file)
        self.patch_dir = patch_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(), # 0-1 range automatically
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Folder se images sequence mein uthana (fire_1.png style)
        patch_list = sorted([f for f in os.listdir(self.patch_dir) if f.endswith('.png')])
        img_path = os.path.join(self.patch_dir, patch_list[idx % len(patch_list)])
        
        img = cv2.imread(img_path, 0)
        if img is None: img = np.zeros((256, 256), dtype=np.uint8)
        img = self.transform(img)

        # Wind features (Sin/Cos normalization)
        w_speed = self.df.iloc[idx]['wind_speed'] / 100.0 # Scale to small value
        w_dir = self.df.iloc[idx]['wind_direction']
        rad = np.deg2rad(w_dir)
        wind_vec = torch.tensor([w_speed * np.cos(rad), w_speed * np.sin(rad)], dtype=torch.float32)
        
        return img, wind_vec

# 2. Architecture
class ConditionalUNet(nn.Module):
    def __init__(self):
        super(ConditionalUNet, self).__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.wind_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.dec1 = nn.ConvTranspose2d(128, 1, 2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x, w):
        x1 = self.relu(self.enc1(x))
        x_small = self.pool(x1)
        w_feat = self.wind_mlp(w).view(-1, 64, 1, 1).expand(-1, 64, 128, 128)
        combined = torch.cat([x_small, w_feat], dim=1)
        out = torch.sigmoid(self.dec1(combined))
        return out

def train_conditional(epochs=50):
    dataset = ConditionalWildfireDataset('data/processed/metadata_with_weather.csv', 'data/processed/patches')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = ConditionalUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()

    print(f"🚀 Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for images, winds in loader:
            optimizer.zero_grad()
            outputs = model(images, winds)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    
    return model

if __name__ == "__main__":
    trained_model = train_conditional(epochs=50) # Epochs barha diye hain
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), "models/conditional_wildfire_model.pth")
    print("✅ Model Saved Successfully!")