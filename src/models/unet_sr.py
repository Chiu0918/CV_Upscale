import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # padding=1 確保卷積後尺寸不變，這對 U-Net 拼接很重要
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetSR(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetSR, self).__init__()
        
        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) # 加深一層，效果更好
        
        # --- Decoder ---
        # 這裡改用 Bilinear 插值上採樣，比 ConvTranspose2d 更平滑，比較不會有棋盤格假影
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(512 + 256, 256)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(256 + 128, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(128 + 64, 64)
        
        # --- Output ---
        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        # 1. 預先放大：如果輸入是 64x64，先放大到 256x256
        if x.size(2) < 256:
             x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)

        # 2. Encoder Path
        x1 = self.inc(x)        # 256 -> 64 ch
        x2 = self.down1(x1)     # 128 -> 128 ch
        x3 = self.down2(x2)     # 64  -> 256 ch
        x4 = self.down3(x3)     # 32  -> 512 ch (Bottleneck)

        # 3. Decoder Path (Skip Connections)
        # 上採樣 -> 拼接 (Cat) -> 卷積
        
        x = self.up1(x4)                    # 32 -> 64 size
        # 處理可能因奇數尺寸導致的大小不合 (雖然這裡都是偶數，但加了保險)
        if x.size(2) != x3.size(2):
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)       # 256 + 512 = 768 ch ?? No, upsample doesn't change ch. 
                                            # up1 output is 512 ch (same as input x4 because Upsample layer has no weights)
                                            # Wait, Upsample just changes size. So x is 512ch.
                                            # Cat with x3 (256ch) -> 778 ch.
                                            # My DoubleConv input needs to match.
        x = self.conv_up1(x)

        x = self.up2(x)                     # 64 -> 128 size
        if x.size(2) != x2.size(2):
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)                     # 128 -> 256 size
        if x.size(2) != x1.size(2):
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)

        # 4. Output
        logits = self.outc(x)
        return logits