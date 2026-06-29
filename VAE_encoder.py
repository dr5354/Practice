import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1 / math.sqrt(hidden)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        return x + self.gamma * out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            Swish(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
        )
        self.act = Swish()

    def forward(self, x):
        return self.act(x + self.block(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            Swish(),
            ResidualBlock(out_ch)
        )
    def forward(self, x):
        return self.block(x)

class VAE_Encoder(nn.Module):
    """
    Энкодер, построенный на твоих кастомных блоках.
    Извлекает латентный вектор стиля (z) из реального изображения.
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.enc1 = DownBlock(in_channels, 64)     # 128 -> 64
        self.enc2 = DownBlock(64, 128)             # 64 -> 32
        self.enc3 = DownBlock(128, 256)            # 32 -> 16
        self.enc4 = DownBlock(256, 512)            # 16 -> 8
        self.attn = SelfAttention(512)
        self.enc5 = DownBlock(512, 512)            # 8 -> 4

        flat_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.attn(x)
        x = self.enc5(x)

        flat = torch.flatten(x, 1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        # Ограничиваем дисперсию для стабильности, как в твоем коде
        logvar = torch.clamp(logvar, -20, 15)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std