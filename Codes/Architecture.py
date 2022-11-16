import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), # 2x192x256 --> 16x192x256
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x192x256 --> 16x96x128
            nn.Conv2d(16, 8, 3, padding=1), # 16x96x128 --> 8x96x128
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x96x128 --> 8x48x64
            nn.Conv2d(8, 4, 3, padding=1), # 8x48x64 --> 4x48x64
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 4x48x64 --> 4x24x32
            nn.Conv2d(4, 2, 3, padding=1), # 4x24x32 --> 2x24x32
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 2x24x32 --> 2x12x16
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # 2x12x16 --> 2x24x32
            nn.Conv2d(2, 4, 3, padding=1), # 2x24x32 --> 4x24x32
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 4x24x32 --> 4x48x64
            nn.Conv2d(4, 8, 3, padding=1), # 4x48x64 --> 8x48x64
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 8x48x64 --> 8x96x128
            nn.Conv2d(8, 16, 3, padding=1), # 8x96x128 --> 16x96x128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 16x96x128 --> 16x192x256
            nn.Conv2d(16, 2, 3, padding=1), # 16x192x256 --> 2x192x256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x