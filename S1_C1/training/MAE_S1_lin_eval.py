# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import random
import os

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.models.layers import trunc_normal_

import MAE.models_vit as models_vit
from MAE.pos_embed import interpolate_pos_embed
from MAE.utils import LARS, NativeScalerWithGradNormCount as NativeScaler, RandomResizedCrop
from MAE.engine_finetune import train_one_epoch, evaluate
from S1_C1.configs.config import load_global_config
from S1_C1.utils.utils import calculate_sampler_weights

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

    transform_train = transforms.Compose([
            # RandomResizedCrop(224, interpolation=3),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std)])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std)])
    transform_c1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std),
            ])
    
    dataset_train = datasets.ImageFolder(config.data.train_path, transform=transform_train)
    dataset_val = datasets.ImageFolder(config.data.val_path, transform=transform_val)
    dataset_c1 = datasets.ImageFolder(config.data.c1_path, transform=transform_c1)

    # Oversampling
    sample_weights = calculate_sampler_weights(dataset_train)
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataloader = DataLoader(dataset_train,
                              sampler=sampler,
                              batch_size=config.train.batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    val_dataloader = DataLoader(dataset_val,
                              batch_size=config.train.val_batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    c1_dataloader = DataLoader(dataset_c1,
                               batch_size=config.train.c1_batch_size,
                               num_workers=config.train.num_workers,
                               pin_memory=config.train.pin_memory,
                               drop_last=False)
    
    model = models_vit.__dict__[config.model.model](num_classes=config.data.num_classes, global_pool=config.model.global_pool)

    checkpoint = torch.load(config.model.checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % config.model.checkpoint_path)
    checkpoint_model = checkpoint['MAE_encoder']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if config.model.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    device = torch.device(config.train.device)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    config.train.lr = config.train.blr * config.train.batch_size / 256
    print("base lr: %.2e" % (config.train.lr * 256 / config.train.batch_size))
    print("actual lr: %.2e" % config.train.lr)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Start training for {config.train.epochs} epochs")

    max_accuracy = 0.0
    for epoch in range(0, config.train.epochs):
        
        train_loss = train_one_epoch(
            model, criterion, train_dataloader,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=None,
            config=config)
        
        val_acc, val_loss, v_counts = evaluate(val_dataloader, model, device)
        c1_acc, c1_loss, c1_counts = evaluate(c1_dataloader, model, device)

        val_loss = val_loss / v_counts
        c1_loss = c1_loss / c1_counts
        acc = val_acc / v_counts
        c1_acc = c1_acc / c1_counts
        print(f'Epoch {epoch} of {config.train.epochs}')
        print(f'train loss is {train_loss}')
        print(f'val loss is {val_loss}')
        print(f'C1 loss is {c1_loss}')
        print(f'validation accuracy is {acc}')
        print(f'C1 accuracy is {c1_acc}')


if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_global_config('configs/MAE_S1_lin_eval.toml')
    main(config)