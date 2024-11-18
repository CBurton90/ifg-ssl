# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import os
import random

# pytorch imports
import torch
import torchvision.transforms as transforms

from utils.config import load_global_config
from hephaestus.dataset.Dataset import FullFrameDataset 

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set:
    torch.backends.cudnn.deterministic = True # Use deterministic algorithms.
    torch.backends.cudnn.benchmark = False # Causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(config: dict) -> None:

    # Set random seed and device
    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed()
    device = torch.device(config.train.device)
    
    train_dataset = FullFrameDataset(config, mode="test", transform=None)

if __name__ == '__main__':
    config = load_global_config('configs/MAE_linear_train-eval_hephaestus.toml')
    main(config)