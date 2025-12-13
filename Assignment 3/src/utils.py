import torch
import os

from modules.models import Generator, Discriminator
from modules.config import CFG

def import_pretrained_generator(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        generator = Generator(CFG.nc, CFG.nz, CFG.ngf)
        generator.load_state_dict(checkpoint)
        
        return generator
    else:
        print(f"Checkpoint path is incorrect: {checkpoint_path}")
        
def import_pretrained_discriminator(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        discriminator = Discriminator(CFG.nc, CFG.ndf)
        discriminator.load_state_dict(checkpoint)
        
        return discriminator
    else:
        print(f"Checkpoint path is incorrect: {checkpoint_path}")