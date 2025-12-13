import sys
sys.path.append("..")

from dotenv import load_dotenv
import os

from utils import import_pretrained_generator
from modules import DDPMConfigs

load_dotenv()

def test_generator_pretrained_import():
    try:
        G = import_pretrained_generator(checkpoint_path=os.getenv('GENERATOR_CHECKPOINT_PATH'))
    except Exception as ex:
        print(f"Unexpected exception: {ex}")
        return False
    else:
        return True
    
def test_ddpm_pretrained_import():
    try:
        # Create configurations
        ddpm = DDPMConfigs()
        # Initialize
        ddpm.init(checkpoint_path=os.getenv('DDPM_CHECKPOINT_PATH'))
    except Exception as ex:
        print(f"Unexpected exception: {ex}")
        return False
    else:
        return True