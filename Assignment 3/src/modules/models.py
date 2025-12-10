import torch.nn as nn
import torch
import torch.nn.functional as F

from .spectral_norm import SpectralNorm
from .unet import UNet
from .utils import get_beta_schedule, extract

# Generator
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Вход: nz x 1 x 1
            
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Размер: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Размер: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Размер: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Размер: ngf x 32 x 32

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Выход в диапазоне [-1, 1]
            # Размер: nc x 64 x 64

        )

    def forward(self, input):
        return self.main(input)

# Discriminator

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Вход: nc x 64 x 64

            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Размер: ndf x 32 x 32

            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Размер: (ndf*2) x 16 x 16

            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Размер: (ndf*4) x 8 x 8

            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Размер: (ndf*8) x 4 x 4

            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()  # Выход: вероятность "настоящести" изображения
            # Размер: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)
    

# Класс DDPM для обучения и сэмплирования
class DDPM(nn.Module):
    """
    Класс для обучения и генерации с помощью DDPM
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UNet(config)

        # Предвычисляем параметры диффузии
        self.timesteps = config.timesteps
        self.betas = get_beta_schedule(config.timesteps, config.beta_start, config.beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)


        # Параметры для обратного процесса
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x0, t, noise=None):
        """
        Прямой процесс диффузии: добавление шума к изображению
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape).to(x0.device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape).to(x0.device)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x0):
        """
        Обучение: предсказываем шум для случайного временного шага
        """
        device = x0.device
        batch_size = x0.shape[0]
        
        # Случайный временной шаг
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Добавляем шум
        noise = torch.randn_like(x0)
        x_noisy = self.forward_diffusion(x0, t, noise)

        # Предсказываем шум
        noise_pred = self.model(x_noisy, t)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, batch_size, device):
        """
        Генерация изображений путем постепенного удаления шума
        """
        self.model.eval()

        # Начинаем с чистого шума
        x = torch.randn(batch_size, self.config.in_channels, 
                       self.config.img_size, self.config.img_size).to(device)

        
        # Постепенно удаляем шум
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
            # Предсказываем шум
            noise_pred = self.model(x, t)
        
            # Параметры для обратного шага
            alpha_t = extract(self.alphas, t, x.shape).to(device)
            alpha_cumprod_t = extract(self.alphas_cumprod, t, x.shape).to(device)
            beta_t = extract(self.betas, t, x.shape).to(device)
            sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape).to(device)
            sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape).to(device)


            # Обратный шаг
            model_mean = sqrt_recip_alpha_t * (
                x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
            )

            if i > 0:
                noise = torch.randn_like(x)
                posterior_variance_t = extract(self.posterior_variance, t, x.shape).to(device)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean

        self.model.train()
        return x