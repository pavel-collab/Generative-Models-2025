import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal Position Embeddings для временного шага
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Позиционное кодирование для временного шага
    Помогает модели понять, на каком этапе диффузии мы находимся
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Residual Block с temporal embeddings
class ResidualBlock(nn.Module):
    """
    Residual блок с добавлением временной информации
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)

        # Добавляем временное кодирование
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

# Attention Block для улучшения качества
class AttentionBlock(nn.Module):
    """
    Self-attention блок для захвата дальних зависимостей
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape для attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        # Scaled dot-product attention
        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)

        return x + h

# U-Net архитектура для DDPM
class UNet(nn.Module):
    """
    U-Net с temporal embeddings для предсказания шума
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Временное кодирование
        time_emb_dim = config.model_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model_channels),
            nn.Linear(config.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Начальная свертка
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList()
        channels = [config.model_channels]
        now_channels = config.model_channels

        for i, mult in enumerate(config.channel_mult):
            out_channels = config.model_channels * mult
            for _ in range(config.num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels, out_channels, time_emb_dim, config.dropout
                ))
                now_channels = out_channels
                channels.append(now_channels)

                # Добавляем attention на нужных разрешениях
                if config.img_size // (2 ** i) in config.attention_resolutions:
                    self.downs.append(AttentionBlock(now_channels))

            # Downsampling (кроме последнего уровня)
            if i != len(config.channel_mult) - 1:
                self.downs.append(nn.Conv2d(now_channels, now_channels, 3, 2, 1))
                channels.append(now_channels)

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, config.dropout),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim, config.dropout)
        ])

        # Upsampling
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(config.channel_mult)):
            out_channels = config.model_channels * mult
            for j in range(config.num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    now_channels + channels.pop(), out_channels, time_emb_dim, config.dropout
                ))
                now_channels = out_channels

            
                # Добавляем attention
                if config.img_size // (2 ** (len(config.channel_mult) - 1 - i)) in config.attention_resolutions:
                    self.ups.append(AttentionBlock(now_channels))

            # Upsampling (кроме последнего уровня)
            if i != len(config.channel_mult) - 1:
                self.ups.append(nn.ConvTranspose2d(now_channels, now_channels, 4, 2, 1))

        # Выходной слой
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, config.in_channels, 3, padding=1)
        )
        
    def forward(self, x, timesteps):
        # Временное кодирование
        time_emb = self.time_mlp(timesteps)

        # Начальная свертка
        h = self.conv_in(x)
        hs = [h]

        # Downsampling
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
            hs.append(h)

        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Upsampling
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, time_emb)
            elif isinstance(layer, AttentionBlock):
                h = layer(h)
            else:  # Upsampling
                h = layer(h)

        # Выход
        return self.conv_out(h)