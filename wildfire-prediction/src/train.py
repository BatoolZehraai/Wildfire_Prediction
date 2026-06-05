import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os
import numpy as np

# 1. Dataset Class with Augmentation
class WildfireDataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_dir = patch_dir
        self.files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
        
        # Augmentation: 139 patches ko multiply karne ke liye
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(), # Normalizes to [0, 1]
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.patch_dir, self.files[idx])
        img = cv2.imread(img_path, 0) # Grayscale read
        if img is None:
            # Fallback agar koi image corrupt ho
            return torch.zeros((1, 256, 256))
        img = self.transform(img)
        return img

# 2. Simple U-Net Architecture (Diffusion Base)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.pool(x)
        x = self.dec1(x)
        return self.sigmoid(x)

# 3. Training Function
def start_training():
    # Folder check
    if not os.path.exists('models'):
        os.makedirs('models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # Data Load karein
    dataset = WildfireDataset('data/processed/patches')
    if len(dataset) == 0:
        print("❌ No patches found! Run generate_patches.py first.")
        return

    # Batch size 4 for 8GB RAM
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = SimpleUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"🔥 Starting Training with {len(dataset)} patches...")
    
    num_epochs = 20 # Aap isay barha sakte hain
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Model save karein
    torch.save(model.state_dict(), "models/wildfire_diffusion_v1.pth")
    print("✅ Training Complete! Model saved in 'models/wildfire_diffusion_v1.pth'")

if __name__ == "__main__":
    start_training()