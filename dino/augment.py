from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import v2

import dino.utils as utils

class ImageAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #imagenet mean's + std's
            ])
        
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224 crop
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
            ])
        
         # second global crop (with solarization)
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224 crop
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
            ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC), #96 crop
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
            ])
        
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
class IfgAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, mean, std):
        flip_and_elastictf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ElasticTransform(alpha=100.0),
            # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std), #Hesphaestus Interferogram means + std's for train dataset mean: tensor([0.6864, 0.5986, 0.5747]) std: tensor([0.3004, 0.2943, 0.2855])
            ])
        
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224 crop
            flip_and_elastictf,
            utils.GaussianBlur(1.0),
            normalize,
            ])
        
         # second global crop (with Gaussian Blur + Noise)
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224 crop
            flip_and_elastictf,
            utils.GaussianBlur(0.1),
            # v2.GaussianNoise(sigma=0.1),
            normalize,
            ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC), #96 crop
            flip_and_elastictf,
            utils.GaussianBlur(p=0.5),
            normalize,
            ])
        
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


