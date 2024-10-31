# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import random
# import datetime
import numpy as np
# import time
import torch
import torch.backends.cudnn as cudnn
# import json
import os

# from pathlib import Path

from timm.models import create_model

from BEiT.optim_factory import create_optimizer
from BEiT.datasets import build_beit_pretraining_insar_dataset
from BEiT.engine_for_pretraining import train_one_epoch
from BEiT.utils import NativeScalerWithGradNormCount as NativeScaler
from BEiT.utils import create_d_vae, cosine_scheduler, auto_load_model
import BEiT.modeling_pretrain

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

def get_model(config):
    print(f"Creating model: {config.model.model}")
    model = create_model(
        config.model.model,
        pretrained=False,
        drop_path_rate=config.model.drop_path,
        use_shared_rel_pos_bias=config.model.rel_pos_bias,
        use_abs_pos_emb=config.model.abs_pos_emb,
        init_values=config.model.layer_scale_init_value,
    )

    return model






def train_BEiT(config):

    device = torch.device(config.train.device)

    set_seed()
    model = get_model(config)

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    window_size = (config.data.input_size // patch_size[0], config.data.input_size // patch_size[1])
    print("Window size = %s" % str(window_size))

    config.model.window_size = window_size
    config.model.patch_size= patch_size

    # get dataset
    dataset_train = build_beit_pretraining_insar_dataset(config)

    print(f'Patch tranform output is size {dataset_train[0][0][0].shape}')
    print(f'Visual token transform output is size {dataset_train[0][0][1].shape}')
    print(f'Mask position output is size {dataset_train[0][0][2].shape}')

    # prepare discrete vae
    d_vae = create_d_vae(
        weight_path=config.model.discrete_vae_weight_path, d_vae_type=config.model.discrete_vae_type,
        device=device, image_size=config.data.second_input_size)
    
    # print(d_vae)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    optimizer = create_optimizer(config, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    num_training_steps_per_epoch = len(dataset_train) // config.train.batch_size
    lr_schedule_values = cosine_scheduler(
        config.optimizer.lr, config.optimizer.min_lr, config.optimizer.epochs, num_training_steps_per_epoch,
        warmup_epochs=config.optimizer.warmup_epochs, warmup_steps=config.optimizer.warmup_steps,
    )

    config.optimizer.weight_decay_end = None

    if config.optimizer.weight_decay_end is None:
        config.optimizer.weight_decay_end = config.optimizer.weight_decay
    wd_schedule_values = cosine_scheduler(
        config.optimizer.weight_decay, config.optimizer.weight_decay_end, config.optimizer.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    auto_load_model(config=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {config.optimizer.epochs} epochs")
    # start_time = time.time()

    clip_grad = 3.0

    for epoch in range(config.resume.start_epoch, config.optimizer.epochs):

        train_stats = train_one_epoch(
            model, d_vae, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            clip_grad, log_writer=None,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
        )

        if epoch % 5 == 0:
            save_dict = {
                'BEiT_encoder': model.state_dict(),
                }
            print('saving checkpoint')
            torch.save(save_dict, config.model.checkpoint_path)




if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_global_config('configs/BEiT_S1_train.toml')
    train_BEiT(config)
