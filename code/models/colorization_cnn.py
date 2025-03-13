import torch
import torch.nn as nn
import torch.nn.functional as F

class ZhangColorizationNet(nn.Module):
    def __init__(self):
        super(ZhangColorizationNet, self).__init__()
        
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.dec1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.out_layer = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        x = F.relu(self.bottleneck(x))
        
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        
        x = torch.tanh(self.out_layer(x))
        return x
