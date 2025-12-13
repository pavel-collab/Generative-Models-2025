import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dotenv import load_dotenv
import torch

from utils import import_pretrained_generator
from modules import DDPMConfigs

load_dotenv()

def test_generator_image_generation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        G = import_pretrained_generator(checkpoint_path=os.getenv('GENERATOR_CHECKPOINT_PATH')).to(device)
        # Генерируем случайный шум
        z = torch.randn(1, 100, 1, 1).to(device)

        # Генерируем изображение
        img = G(z)[0]

        # Денормализация из [-1, 1] в [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        assert False
    else:
        assert True
    
def test_ddpm_image_geenration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Create configurations
        ddpm = DDPMConfigs()
        # Initialize
        ddpm.init(checkpoint_path=os.getenv('DDPM_CHECKPOINT_PATH'))
        samples = ddpm.sample()
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        assert False
    else:
        assert True
