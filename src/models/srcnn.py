import torch
from torch import nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        """
        SRCNN 模型架構 (Super-Resolution Convolutional Neural Network)
        
        Args:
            num_channels (int): 輸入影像的通道數 (RGB=3)
        """
        super(SRCNN, self).__init__()
        
        # 第一層：特徵提取 (Feature Extraction)
        # 9x9 kernel, 輸入 channels -> 64
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二層：非線性映射 (Non-linear Mapping)
        # 5x5 kernel, 64 -> 32
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # 第三層：重建 (Reconstruction)
        # 5x5 kernel, 32 -> 輸出 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # ---------------------------------------------------------
        # SRCNN 的特點是：它預期輸入的圖片已經是「大圖 (256x256)」了。
        # 但我們的 Dataset 讀進來的 LR 是「小圖 (64x64)」。
        # 所以我們在模型的第一步，先用 Bicubic 插值法幫它放大 4 倍。
        # ---------------------------------------------------------
        
        if x.size(2) < 256: # 簡單判斷：如果高度小於 256，代表是小圖
             x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        
        # 通過三層網路
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        
        return x