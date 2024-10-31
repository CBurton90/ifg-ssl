import sys
sys.path.append("/home/conradb/git/ifg-ssl")
from collections import defaultdict, Counter, OrderedDict
#from sklearn.model_selection import train_test_split
import numpy as np
#import matplotlib.pyplot as plt

# pytorch
import torch
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# local module imports
from S1_C1.configs.config import load_global_config
from S1_C1.utils.utils import calculate_sampler_weights
import dino.vision_transformer as vit
from dino.linear_classifier import LinearClassifier, train, validate
import dino.utils as utils

def train_linear(config):
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.ElasticTransform(alpha=100.0),
        #utils.GaussianBlur(p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
        ])
    
    c1_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
        ])
    
    train_dataset = datasets.ImageFolder(root=config.data.train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.data.val_path, transform=val_transform)
    c1_dataset = datasets.ImageFolder(root=config.data.c1_val_path, transform=c1_transform)
    
    sample_weights = calculate_sampler_weights(train_dataset)
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataloader = DataLoader(train_dataset,
                              sampler=sampler,
                              batch_size=config.train.batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    val_dataloader = DataLoader(val_dataset,
                              batch_size=config.train.val_batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=False)
    
    c1_dataloader = DataLoader(c1_dataset,
                               batch_size=config.train.c1_val_batch_size,
                               num_workers=config.train.num_workers,
                               pin_memory=config.train.pin_memory,
                               drop_last=True)
    
    idx = torch.randint(63, (1,))
    print(idx)
    #plt.imsave(fname='outputs/train_aug_example_'+str(idx.item())+'.png', arr=train_dataset[idx.item()][0].clamp(0,1).numpy().transpose(1,2,0), format='png')
    #plt.imsave(fname='outputs/val_aug_example_'+str(idx.item())+'.png', arr=val_dataset[idx.item()][0].clamp(0,1).numpy().transpose(1,2,0), format='png')
    #plt.imsave(fname='outputs/C1_aug_example_'+str(idx.item())+'.png', arr=c1_dataset[idx.item()][0].clamp(0,1).numpy().transpose(1,2,0), format='png')

    if config.model.model in vit.__dict__.keys():
        arch = 'vit'
        model = vit.__dict__[config.model.model](patch_size=config.model.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (config.linear.n_last_blocks + int(config.linear.avgpool_patchtokens))
    elif config.model.model in torchvision_models.__dict__.keys():
        arch = 'resnet50'
        model = torchvision_models.__dict__[config.model.model]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = torch.nn.Identity()
    else:
        print('Model not available')

    state_dict = torch.load(config.model.checkpoint_path, map_location="cpu")
    state_dict = state_dict['teacher']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

    linear_classifier = LinearClassifier(embed_dim, num_labels=config.linear.num_labels)

    # set optimizer
    optimizer = torch.optim.SGD(linear_classifier.parameters(),
                            config.linear.lr,
                            momentum=0.9,
                            weight_decay=0, # we do not apply weight decay 
                            )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs, eta_min=0)

    for epoch in range(0, config.train.epochs):
        total_loss, t_counts = train(model, linear_classifier, optimizer, train_dataloader, epoch, config.linear.n_last_blocks, config.linear.avgpool_patchtokens, config.train.device, arch=arch)
        val_acc, val_loss, v_counts = validate(val_dataloader, model, linear_classifier, config.linear.n_last_blocks, config.linear.avgpool_patchtokens, config.train.device, arch=arch)
        c1_acc, c1_loss, c1_counts = validate(c1_dataloader, model, linear_classifier, config.linear.n_last_blocks, config.linear.avgpool_patchtokens, config.train.device, arch=arch)
        scheduler.step()

        epoch_loss = total_loss / t_counts
        val_loss = val_loss / v_counts
        c1_loss = c1_loss / c1_counts
        acc = val_acc / v_counts
        c1_acc = c1_acc / c1_counts
        print(f'Epoch {epoch} of {config.train.epochs}')
        print(f'train loss is {epoch_loss}')
        print(f'val loss is {val_loss}')
        print(f'C1 loss is {c1_loss}')
        print(f'validation accuracy is {acc}')
        print(f'C1 accuracy is {c1_acc}')

if __name__ == '__main__':
    config = load_global_config('configs/dino_S1_train.toml')
    train_linear(config)

