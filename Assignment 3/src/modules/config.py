import torch

class CFG:
    batch_size = 128
    num_epochs = 300
    workers = 4
    seed = 2021
    image_size = 64
    download = True
    dataroot = "data"
    nc = 3  ## number of chanels
    ngf = 64  # Size of feature maps in generator
    nz = 100  # latent random input vector
    ndf = 64  # Size of feature maps in discriminator
    lr = 0.0002
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_dir = "./images/"
    
# Конфигурация для DDPM
class DDPMConfig:
    timesteps = 1000  # Количество шагов диффузии
    beta_start = 0.0001  # Начальное значение beta
    beta_end = 0.02  # Конечное значение beta
    img_size = 64
    in_channels = 3
    model_channels = 128  # Базовое количество каналов
    num_res_blocks = 2  # Количество residual блоков
    attention_resolutions = [16, 8]  # Разрешения для self-attention
    channel_mult = (1, 2, 2, 4)  # Множители каналов на разных уровнях
    dropout = 0.1
    lr = 2e-4
    batch_size = 64
    num_epochs = 300