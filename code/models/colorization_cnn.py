import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetColorizationNet(nn.Module):
    def __init__(self):
        super(UNetColorizationNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 512)

        # Decoder
        self.dec1 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec3 = self.upconv_block(128, 64)

        # Output layer (2 channels for a,b color space)
        self.out_layer = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)

        # Output layer
        out = self.out_layer(d3)
        return out
