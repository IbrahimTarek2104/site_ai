import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive Conv3x3 -> BatchNorm -> ReLU blocks with Spatial Dropout."""
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.block(x)

class Down(nn.Module):
    """MaxPool2x2 followed by a DoubleConv block."""
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(in_ch, out_ch, drop)
        )
    def forward(self, x): 
        return self.net(x)

class Up(nn.Module):
    """Bilinear upsample -> concatenate attention-weighted skip connection -> DoubleConv."""
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, drop)

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
        x = F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])
        return self.conv(torch.cat([skip, x], dim=1))

class AttentionGate(nn.Module):
    """Soft spatial attention gate to suppress background noise."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        alpha = self.psi(F.relu(g1 + x1, inplace=True))
        return x * alpha

class AttentionUNet(nn.Module):
    """Attention U-Net configured for 3-channel environmental state space mapping."""
    def __init__(self, in_channels=3, base=64, drop=0.2):
        super().__init__()
        f = base
        self.e1 = DoubleConv(in_channels, f, drop)
        self.e2 = Down(f, f*2, drop)
        self.e3 = Down(f*2, f*4, drop)
        self.e4 = Down(f*4, f*8, drop)
        self.bot = Down(f*8, f*16, drop)

        self.a4 = AttentionGate(f*16, f*8, f*4)
        self.a3 = AttentionGate(f*8, f*4, f*2)
        self.a2 = AttentionGate(f*4, f*2, f)
        self.a1 = AttentionGate(f*2, f, f//2)

        self.d4 = Up(f*16 + f*8, f*8, drop)
        self.d3 = Up(f*8 + f*4, f*4, drop)
        self.d2 = Up(f*4 + f*2, f*2, drop)
        self.d1 = Up(f*2 + f, f, drop)
        self.out = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        bn = self.bot(s4)
        d4 = self.d4(bn, self.a4(g=bn, x=s4))
        d3 = self.d3(d4, self.a3(g=d4, x=s3))
        d2 = self.d2(d3, self.a2(g=d3, x=s2))
        d1 = self.d1(d2, self.a1(g=d2, x=s1))
        return torch.sigmoid(self.out(d1))