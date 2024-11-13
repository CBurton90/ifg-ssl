import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torchvision.transforms as transforms

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

if __name__ == '__main__':
    main()
