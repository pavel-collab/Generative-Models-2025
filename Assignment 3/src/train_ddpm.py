import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from modules import DDPMConfigs, CFG

plt.style.use("ggplot")

# Create configurations
configs = DDPMConfigs()

# Initialize
configs.init()

# Start and run the training loop
configs.run()

print("\nОбучение и генерация DDPM завершены!")

MODEL_SAVE_PATH = './models'
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

torch.save(configs.eps_model.state_dict(), f'{MODEL_SAVE_PATH}/ddpm_unet.pth')
