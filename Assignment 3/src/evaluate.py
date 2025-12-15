import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from dotenv import load_dotenv
import os

from utils import run_full_evaluation, import_pretrained_generator
from modules import CFG, DDPMConfigs

load_dotenv()

cifar_dataset = CIFAR10(
            root=CFG.dataroot,
            download=CFG.download,
            transform=transforms.Compose(
                [
                    transforms.Resize([CFG.image_size, CFG.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

'''
Load generator on cpu, because pytorch_image_generation_metrics using multiprocessing,
when Linux fork the process it inherit the cuda context and can trigger an Error. Because of that we're
loading GAN on cpu.
'''
G = import_pretrained_generator(checkpoint_path=os.getenv('GENERATOR_CHECKPOINT_PATH')).to('cpu')

# Create configurations
ddpm = DDPMConfigs()
# Initialize
ddpm.init(checkpoint_path=os.getenv('DDPM_CHECKPOINT_PATH'))

# Запуск оценки
results, df = run_full_evaluation(
    G=G,
    z_dim=100,
    ddpm=ddpm,
    cifar_dataset=cifar_dataset,
    num_samples=5000,
    device=device
)
