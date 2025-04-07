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
import timm.optim.optim_factory as optim_factory

# DL stack imports
# import wandb
import webdataset as wds

# local imports
from utils.config import load_global_config
from hephaestus.dataset.Dataset import HephaestusCompleteDataset
import MAE.models_mae as models_mae
from MAE.utils import NativeScalerWithGradNormCount as NativeScaler, load_model
from MAE.engine_pretrain import train_one_epoch

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
    # Web Dataset does not like instance class string in sample dict when decoding, remove as unecessary for SSL pretraining
    sample.pop("cls")
    return sample

def main(model: torch.nn.Module, config: dict) -> None:
    # Set random seed and device
    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed()
    device = torch.device(config.train.device)

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

    ifg_transform = transforms.Compose([
        # transforms.Resize(config.data.input_size),
        transforms.RandomResizedCrop(config.data.input_size, scale=(0.2, 1.0), interpolation=3), # Bountos et al uses RandomCrop
        # transforms.RandomCrop(config.data.input_size),
        transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=config.data.ifg_mean, std=config.data.ifg_std)
    ])

    cc_transform = transforms.Compose([
        # transforms.Resize(config.data.input_size),
        transforms.RandomResizedCrop(config.data.input_size, scale=(0.2, 1.0), interpolation=3), # Bountos et al uses RandomCrop
        # transforms.RandomCrop(config.data.input_size),
        transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=config.data.cc_mean, std=config.data.cc_std)
    ])

    if config.train.use_wds:
        print('Using WebDataset (sequential sharded I/O)')
        train_dataset = wds.DataPipeline(
            wds.SimpleShardList(os.path.join(config.data.train_path, "hephaestus-{000000..000112}.tar")),
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
            wds.map(pop_class),
            wds.decode("torch"),
            wds.to_tuple("ifg.png", "cc.png"),
            wds.map_tuple(ifg_transform, cc_transform),
            wds.batched(config.train.batch_size),
        )
        
        train_loader = wds.WebLoader(
            train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=config.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
            )
    else:
        print('Using PyTorch Dataset/Dataloader')
        #TODO Add standard PyTorch Dataset + Dataloader

        transform = transforms.Compose([
            # transforms.Resize(config.data.input_size),
            # transforms.RandomResizedCrop(config.data.input_size, scale=(0.15, 1.0), interpolation=3), # Bountos et al uses RandomCrop
            transforms.RandomCrop(config.data.input_size),
            transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.ifg_mean, std=config.data.ifg_std)
            ])
        train_dataset = HephaestusCompleteDataset(config, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
        batch = next(iter(train_dataset))
        plt.imsave('tmp/ifg_first_samp.png', batch.clamp(0,1).numpy().transpose(1,2,0))
        print('printing batch')
        print(batch)
        print(torch.min(batch), torch.max(batch))


    # batch = next(iter(train_dataset))

    # # idx = torch.randint(9, (1,))
    # idx=torch.tensor(4)
    # print(idx)
    # plt.imsave('tmp/ifg_'+str(idx.item())+'.png', batch[0][idx.item()].clamp(0,1).numpy().transpose(1,2,0))
    # plt.imsave('tmp/cc_'+str(idx.item())+'.png', batch[1][idx.item()].clamp(0,1).numpy().transpose(1,2,0))
    # print(batch[0][idx.item()])
    # print(torch.min(batch[0][idx.item()][0]), torch.max(batch[0][idx.item()][0]))
    # print(batch[1][idx.item()])
    # print(torch.min(batch[1][idx.item()][0]), torch.max(batch[1][idx.item()][0]))
    

    # Scale learning rate, we are using 1 GPU (not distributed training) therefore not really necessary
    config.train.lr = config.train.blr * config.train.batch_size / 256
    print("base lr: %.2e" % (config.train.lr * 256 / config.train.batch_size))
    print("actual lr: %.2e" % config.train.lr)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, config.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    load_model(config, model, optimizer, loss_scaler)

    print(f"Start training for {config.train.epochs} epochs")

    for epoch in range(0, config.train.epochs):
        train_stats, lr = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            config=config
        )

        print(f'Epoch {epoch}/{config.train.epochs}, Batch Loss is {train_stats}, LR is {lr}')

        if epoch % 5 == 0:
            save_dict = {
                'epoch': epoch,
                'arch' : config.model.model,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            print('saving checkpoint')
            torch.save(save_dict, config.model.checkpoint_save_path)


        


if __name__ == '__main__':
    config = load_global_config('configs/MAE_pretraining_hephaestus.toml')
    model = create_model(config)
    main(model, config)
