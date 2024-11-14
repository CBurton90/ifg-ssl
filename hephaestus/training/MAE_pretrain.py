# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# pytorch imports
import torch
import torchvision.transforms as transforms

# DL stack imports
# import wandb
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

def pop_class(sample: dict) -> None:
    sample.pop("cls")
    return sample

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

    base_transform = transforms.Compose([
        transforms.Resize(config.data.input_size),
        transforms.RandomCrop(config.data.input_size),
        transforms.Grayscale(num_output_channels=3)
    ])

    if config.train.use_wds:
        print('Using WebDataset (sequential sharded I/O)')
        train_dataset = wds.DataPipeline(
            wds.SimpleShardList(os.path.join(config.data.train_path, "hephaestus-{000000..000002}.tar")),
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
            wds.map(pop_class),
            wds.decode("torch"),
            wds.to_tuple("ifg.png", "cc.png"),
            wds.map_tuple(base_transform, base_transform),
            wds.batched(config.train.batch_size),
        )
        # train_dataset = wds.WebDataset(config.data.train_path).shuffle(1000)
    else:
        print('Using PyTorch Dataset/Dataloader')

    batch = next(iter(train_dataset))

    idx = torch.randint(9, (1,))
    print(idx)
    plt.imsave('tmp/ifg_'+str(idx)+'.png', batch[0][idx.item()].numpy().transpose(1,2,0))
    plt.imsave('tmp/cc_'+str(idx)+'.png', batch[1][idx.item()].numpy().transpose(1,2,0))
        


if __name__ == '__main__':
    config = load_global_config('configs/MAE_pretraining_hephaestus.toml')
    model = create_model(config)
    main(model, config)
