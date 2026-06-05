import torch
import torch.nn as nn

class WildfireDiffusionModel(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(WildfireDiffusionModel, self).__init__()
        # 5 Channels: Fire(t-1), Temp, Wind_X, Wind_Y, Humidity
        # Output: Predicted Fire(t)
        self.unet = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        print("Wildfire Diffusion Model Initialized by Ahmer Ali")

    def forward(self, x):
        return self.unet(x)