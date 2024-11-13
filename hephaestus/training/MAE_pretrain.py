# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import os
import random

# pytorch imports
import torch

# DL stack imports
import wandb
import webdataset as wds

# local imports
from utils.config import load_global_config
import MAE.models_mae as models_mae





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

def create_model(config: dict) -> torch.nn.Module:
    model = models_mae.__dict__[config.model.model](norm_pix_loss=config.model.norm_pix_loss)
    return model

def main(model: torch.nn.Module, config: dict) -> None:
    # Set random seed and device
    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed()
    device = torch.device(config.train.device)
    print(config)

    # Load model
    model = create_model(config)
    model.to(device)

    # Initialize WandB
    # id = wandb.sdk.lib.runid.generate_id()
    # config.logging.wandb_id = id
    # wandb.init(
    #     project=config.logging.wandb_project,
    #     entity=config.logging.wandb_entity,
    #     config={
    #         "learning_rate": config.train.lr,
    #         "epochs": config.train.epochs
    #     },
    #     id=id,
    # )
    # wandb.watch(model)

    if config.train.use_wds:
        print('Using WebDataset (sequential sharded I/O)')
    else:
        print('Using PyTorch Dataset/Dataloader')
        


if __name__ == '__main__':
    config = load_global_config('configs/MAE_pretraining_hephaestus.toml')
    main(config)
