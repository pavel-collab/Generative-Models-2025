import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from dotenv import load_dotenv
import os

from utils import run_full_evaluation, import_pretrained_generator
from modules import CFG, DDPMConfigs

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

G = import_pretrained_generator(checkpoint_path=os.getenv('GENERATOR_CHECKPOINT_PATH')).to(device)

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