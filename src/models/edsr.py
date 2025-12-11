import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    """
    沒有 BN 的Residuel
    """
    def __init__(self, n_feats, kernel_size, bias=True, act=nn.ReLU(True), res_scale=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act = act
        self.res_scale = res_scale 

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        res = res.mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    """
    PixelShuffle
    """
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:   
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))
                if act: m.append(act())
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if act: m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, n_channels=3, n_resblocks=16, n_feats=64, scale=4, res_scale=0.1):
        super(EDSR, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        self.head = nn.Conv2d(n_channels, n_feats, kernel_size, padding=(kernel_size//2))
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))
        self.body = nn.Sequential(*m_body)

        m_tail = [
            Upsampler(scale, n_feats, act=False),
            nn.Conv2d(n_feats, n_channels, kernel_size, padding=(kernel_size//2))
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        
        res = self.body(x)
        res += x 
        
        x = self.tail(res)
        return x