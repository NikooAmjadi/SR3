import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, down=True):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1) if down else nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.down = down

    def forward(self, x, t_emb):
        h = self.block(x)
        # Add time embedding (broadcasted to match spatial shape)
        t = self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)
        return h + t


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)

        # Middle
        self.mid = ConvBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        # Decoder
        self.dec1 = ConvBlock(base_channels * 2, base_channels, time_emb_dim, down=False)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoding
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(x1, t_emb)

        # Middle
        x_mid = self.mid(x2, t_emb)

        # Decoding
        x3 = self.dec1(x_mid, t_emb)
        out = self.out_conv(x3)
        return out
