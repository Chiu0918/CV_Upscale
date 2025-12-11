import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
     Residual Connection
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class AttentionBlock(nn.Module):
    """
    Attention Gate
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetSR(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetSR, self).__init__()
        
        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = ResidualBlock(64, 128)
        self.down2 = ResidualBlock(128, 256)
        self.down3 = ResidualBlock(256, 512) 
        
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.conv_up1 = ResidualBlock(768, 256) 
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.conv_up2 = ResidualBlock(384, 128) 

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionBlock(F_g=128, F_l=64, F_int=32)
        self.conv_up3 = ResidualBlock(192, 64)

        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=True)
        
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3)) 
        
        d1 = self.up1(x4)
        if d1.shape[2:] != x3.shape[2:]:
            d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=True)
            
        x3 = self.att1(g=d1, x=x3)
        d1 = torch.cat([x3, d1], dim=1)
        d1 = self.conv_up1(d1)
        
        d2 = self.up2(d1)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
            
        x2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up2(d2)
        
        d3 = self.up3(d2)
        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        x1 = self.att3(g=d3, x=x1)
        d3 = torch.cat([x1, d3], dim=1)
        output = self.conv_up3(d3)
        
        output = self.outc(output)
        return output