import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

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
    
def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for t and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * eps_model - epsilon_theta(x_t, t) model
        * n_steps - t
        * device - the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create beta_1 ... beta_T linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # alpha_t = 1 - beta_t
        self.alpha = 1 - self.beta # YOUR CODE HERE
        # [1]
        self.alpha_bar = torch.cumprod(self.alpha, 0) # YOUR CODE HERE
        # T
        self.n_steps = n_steps
        # sigma^2 = beta
        self.sigma2 = self.beta # YOUR CODE HERE

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(x_t|x_0) distribution

        [2]
        """
        # [3]
        #! выбираем коэффициенты для шага t
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # [4]
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    #! генерируем зашемленное изображение, при этом обуславливаясь на предыдущее зашумленное изображение
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample from q(x_t|x_0)

        [5]
        """

        # [6]
        if eps is None:
            eps = torch.randn_like(x0) # YOUR CODE HERE

        # get q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t) # YOUR CODE HERE
        # Sample from q(x_t|x_0)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_theta(x_{t-1}|x_t)

        [7]
        """

        # epsilon_theta(x_t, t)
        #! Берем нашу обученную модель eps_model (она научилась поредсказывать шум) и подаем ей на вход зашумленное изображение и шаг t
        eps_theta = self.eps_model(xt, t) # YOUR CODE HERE
        # [8]
        #! выбираем коэффициенты для шага t
        alpha_bar = gather(self.alpha_bar, t)
        # alpha_t
        #! выбираем коэффициенты для шага t
        alpha = gather(self.alpha, t)
        # [9]
        eps_coef = (1.0 - alpha) / (1.0 - alpha_bar).sqrt() # YOUR CODE HERE
        # [10]
        #! на вход получили некоторое изображение xt, теперь удаляем из него шум, который получили с помощью обученной модели
        mean = (xt - eps_coef * eps_theta) / alpha.sqrt() # YOUR CODE HERE
        # sigma^2
        var = gather(self.sigma2, t)

        # [11]
        expand_dims = xt.dim() - 1
        #! накидываем маску, чтобы занулить шум там, где t=0
        mask = (t > 0).float().view(-1, *([1] * expand_dims))
        eps = torch.randn_like(xt) * mask # YOUR CODE HERE
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Simplified Loss

        [12]
        """
        # Get batch size
        batch_size = x0.shape[0] # YOUR CODE HERE
        # Get random t for each sample in the batch
        #! На стадии обучения берем случайный t
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long) # YOUR CODE HERE

        # [13]
        if noise is None:
            noise = torch.randn_like(x0) # YOUR CODE HERE

        # Sample x_t for q(x_t|x_0)
        #! Делаем репараметризацию (reparametrsation trick)
        xt = self.q_sample(x0, t, noise) # YOUR CODE HERE
        # [14]
        #! получаем предсказание обучаемой модели
        eps_theta = self.eps_model(xt, t) # YOUR CODE HERE

        # MSE loss
        return F.mse_loss(eps_theta, noise) # YOUR CODE HERE
