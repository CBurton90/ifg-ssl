# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import random
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm.optim.optim_factory as optim_factory

import MAE.models_mae as models_mae
from MAE.utils import NativeScalerWithGradNormCount as NativeScaler
from MAE.engine_pretrain import train_one_epoch
from S1_C1.configs.config import load_global_config

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(config):

    set_seed()
    device = torch.device(config.train.device)

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.data.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std)])
    dataset_train = datasets.ImageFolder(config.data.train_path, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[config.model.model](norm_pix_loss=config.model.norm_pix_loss)
    model.to(device)
    model_without_ddp = model

    config.train.lr = config.train.blr * config.train.batch_size / 256

    print("base lr: %.2e" % (config.train.lr * 256 / config.train.batch_size))
    print("actual lr: %.2e" % config.train.lr)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, config.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    print(f"Start training for {config.train.epochs} epochs")

    for epoch in range(0, config.train.epochs):
        train_stats, lr = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            config=config
        )

        print(f'Epoch {epoch}/{config.train.epochs}, Batch Loss is {train_stats}, LR is {lr}')

        if epoch % 5 == 0:
            save_dict = {
                'MAE_encoder': model.state_dict(),
                }
            print('saving checkpoint')
            torch.save(save_dict, config.model.checkpoint_path)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_global_config('configs/MAE_S1_pretrain.toml')
    main(config)