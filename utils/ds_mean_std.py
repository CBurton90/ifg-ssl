import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torchvision.transforms.v2 as transforms

# local imports
from utils.config import load_global_config
from hephaestus.dataset.Dataset import HephaestusCompleteDataset

def main():

    ####### COMPUTE MEAN / STD

    
    dataset = datasets.ImageFolder('/scratch/SDF25/PrototypeInSAR/synth/', transform=transforms.ToTensor())

    train_dataloader = DataLoader(dataset, batch_size=256, num_workers=32, pin_memory=False, drop_last=True)

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for _, (inputs, _) in tqdm(enumerate(train_dataloader)):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    # pixel count
    count = (len(train_dataloader)*256) * 224 * 224 #n_batches x batch_size x pixel height x pixel width

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))

def hephaestus_mu_std():

    config = load_global_config('../hephaestus/configs/MAE_pretraining_hephaestus.toml')
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    train_dataset = HephaestusCompleteDataset(config, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=32, pin_memory=False, drop_last=False)

    # placeholders
    mu = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    count = 0

    for idx, inputs in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        inp_mu = torch.mean(inputs.squeeze(0), dim=(1,2))
        inp_std = torch.std(inputs.squeeze(0), dim=(1,2))

        mu += inp_mu
        std += inp_std
        count += 1

    total_mu = mu / count
    total_std = std / count

    # output
    print("mean: " + str(total_mu))
    print("std:  " + str(total_std))


if __name__ == '__main__':
    # main()
    hephaestus_mu_std()
