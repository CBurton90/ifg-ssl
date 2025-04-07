import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
from typing import Tuple
ImageFile.LOAD_TRUNCATED_IMAGES=True

import torch
from torch.utils.data import Dataset

# class HephaestusCompleteDataset(torch.utils.Dataset):
#     """
#     Hephaestus Dataset
#     @InProceedings{Bountos_2022_CVPR,
#     author    = {Bountos, Nikolaos Ioannis and Papoutsis, Ioannis and Michail, Dimitrios and Karavias, Andreas and Elias, Panagiotis and Parcharidis, Isaak},
#     title     = {Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
#     month     = {June},
#     year      = {2022},
#     pages     = {1453-1462}
#     """

#     def __init__(self, config, transform=None):

def get_insar_path(annotation_path, root_path="/scratch/SDF25/LiCSAR-web-tools/"):
    reader = open(annotation_path)
    annotation = json.load(reader)
    frameID = annotation["frameID"]
    primaryDate = annotation["primary_date"]
    secondaryDate = annotation["secondary_date"]
    primary_secondary = primaryDate + "_" + secondaryDate
    img_path = (
        root_path
        + frameID
        + "/interferograms/"
        + primary_secondary
        + "/"
        + primary_secondary
        + ".geo.diff.png"
    )
    return img_path


class HephaestusCompleteDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None) -> None:
        self.data_path = config.data.train_path
        self.transform = transform
        self.interferograms = []
        self.channels = 3
        frames = os.listdir(config.data.train_path)
        for frame in tqdm(frames, total=len(frames)):
            frame_path = config.data.train_path + frame + "/interferograms/"
            caption = os.listdir(frame_path)
            for cap in caption:
                caption_path = frame_path + cap + "/" + cap + ".geo.diff.png"

                if os.path.exists(caption_path):
                    image_dict = {"path": caption_path, "frame": frame}
                    self.interferograms.append(image_dict)
                else:
                    continue

        self.num_examples = len(self.interferograms)

    def __len__(self) -> int:
        return self.num_examples

    # def prepare_insar(self, insar):
    #     insar = torch.from_numpy(insar).float().permute(2, 0, 1)
    #     insar /= 255
    #     return insar

    # def read_insar(self, path):
    #     insar = cv.imread(path, 0)
    #     if insar is None:
    #         print("None")
    #         return insar

    #     insar = np.expand_dims(insar, axis=2).repeat(self.channels, axis=2)
    #     transform = self.augmentations(image=insar)
    #     insar_1 = transform["image"]
    #     transform_2 = self.augmentations(image=insar)
    #     insar_2 = transform_2["image"]

    #     insar_1 = self.prepare_insar(insar_1)
    #     insar_2 = self.prepare_insar(insar_2)
    #     return (insar_1, insar_2)

    def load_image(self, idx: int) -> Image.Image:
        sample = self.interferograms[idx]
        path = sample["path"]
        # insar = cv2.imread(path, 0)
        # print(insar.shape)
        # print(path)

        return ImageFile.Image.open(path).convert('RGB')
        # return Image.fromarray(insar)

    def __getitem__(self, idx) -> torch.Tensor:

        try:
            insar = self.load_image(idx)
            
            if self.transform:
                insar = self.transform(insar)
            else:
                insar = insar
            return insar
            
        except (OSError, AttributeError, TypeError) as e:
            print(f'Failed on idx {idx}')

class FullFrameDataset(torch.utils.data.Dataset):
    '''
        Dataloader returning the full InSAR frame.
        Returns InSAR + Coherence , labels.
    '''
    def __init__(self, config, mode="train", transform=None):
        self.data_path = config.data.train_path
        self.oversampling = config.train.oversampling
        self.config = config
        self.mode = mode
        self.transform = transform
        self.interferograms = []
        self.channels = 3
        self.frame_dict = {}
        self.positives = []
        self.negatives = []
        annotation_path = config.data.annotation_path
        annotations_list = os.listdir(annotation_path)
        frames = os.listdir(config.data.train_path)
        unique_frames = np.unique(frames)
        test_frames = config.data.test_frames

        for idx, frame in enumerate(unique_frames):
            self.frame_dict[frame] = idx
        
        for idx, annotation_file in tqdm(enumerate(annotations_list)):
            annotation = json.load(open(annotation_path + annotation_file, "r"))

            if annotation["frameID"] in test_frames and (mode == "train" or mode == "val"):
                continue
            
            if annotation["frameID"] not in test_frames and mode == "test":
                continue

            if "Non_Deformation" in annotation["label"]:
                label = 0
            else:
                label = 1

            sample_insar_path = get_insar_path(annotation_path=annotation_path + annotation_file, root_path=config.data.train_path)
            sample_cc_path = sample_insar_path[:-8] + 'cc.png'

            if not os.path.isfile(sample_cc_path) or not os.path.isfile(sample_insar_path):
                continue

            sample_dict = {"frameID": annotation["frameID"], "insar_path": sample_insar_path, "label": annotation}
            self.interferograms.append(sample_dict)

            if label == 0:
                self.negatives.append(sample_dict)
            else:
                self.positives.append(sample_dict)

        random.Random(999).shuffle(self.negatives)
        random.Random(999).shuffle(self.positives)

        if self.mode == "train":
            self.positives = self.positives[:int(0.9*len(self.positives))]
            self.negatives = self.negatives[:int(0.9*len(self.negatives))]
            self.interferograms = self.positives.copy()
            self.interferograms.extend(self.negatives.copy())
            self.num_examples = len(self.interferograms)

        elif self.mode=='val':
            self.positives = self.positives[int(0.9*len(self.positives)):]
            self.negatives = self.negatives[int(0.9*len(self.negatives)):]
            self.interferograms = self.positives.copy()
            self.interferograms.extend(self.negatives.copy())
            self.num_examples = len(self.interferograms)

        else:
            self.num_examples = len(self.interferograms)

        print('Mode: ',self.mode,' Number of examples: ',self.num_examples)
        print('Number of positives: ',len(self.positives))
        print('Number of negatives: ',len(self.negatives))

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor]:
        insar = None
        if self.oversampling and self.mode == "train":
            while insar is None:
                choice = random.randint(0,1)
                if choice == 0:
                    choice_neg = random.randint(0, len(self.negatives) -1)
                    sample = self.negatives[choice_neg]
                else:
                    choice_pos = random.randint(0, len(self.positives) -1)
                    sample = self.positives[choice_pos]

                #insar = ImageFile.Image.open(sample["insar_path"]).convert('RGB')
                insar = cv2.imread(sample["insar_path"], 0)
                insar = Image.fromarray(insar)

                if self.transform:
                    insar = self.transform(insar)
                else:
                    insar = insar

        else:
            while insar is None:
                sample = self.interferograms[idx]
                insar = ImageFile.Image.open(sample["insar_path"]).convert('RGB')

                if self.transform:
                    insar = self.transform(insar)
                else:
                    insar = insar

                if insar is None:
                    if idx < self.num_examples -1:
                        idx += 1
                    else:
                        idx = 0

        annotation = sample["label"]
        if "Non_Deformation" in annotation["label"]:
            label = torch.tensor(0).long()
        else:
            label = torch.tensor(1).long()

        return insar, label


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train", transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.train_test_split = json.load(open(config.data.split_path, "r"))
        self.valid_files = self.train_test_split[mode]["0"]
        self.valid_files.extend(self.train_test_split[mode]["1"])
        self.records = []
        self.oversampling = config.train.oversampling
        self.positives = []
        self.negatives = []
        self.frameIDs = []
        self.root_path = config.data.cls_path
        self.missing_file_list = []
        self.missing_file_count = 0

        for file in self.valid_files:
            try:
                annotation = json.load(open(self.root_path + file, "r"))
                # print(annotation)
                sample = file.split("/")[-1]
                # print(sample)
                insar_path = (
                    config.data.train_path
                    + "InSARLabelled/"
                    + str(annotation["Deformation"][0])
                    + "/"
                    + sample[:-5]
                    + ".png"
                    )
                # print(insar_path)

                mask_path = (
                    config.data.train_path
                    + "masks/"
                    + str(annotation["Deformation"][0])
                    + "/"
                    + sample[:-5]
                    + ".png"
                )

                if len(annotation["Deformation"]) > 1:
                    for activity in annotation["Deformation"]:
                        insar_path = (
                            config.data.train_path
                            + "InSARLabelled/"
                            + str(activity)
                            + "/"
                            + sample[:-5]
                            + ".png"
                            )
                        mask_path = (
                            config.data.train_path
                            + "/masks"
                            + str(activity)
                            + "/"
                            + sample[:-5]
                            + ".png"
                            )

                        if os.path.isfile(insar_path):
                            break

                record = {
                    "frameID": annotation["frameID"],
                    "label": annotation["Deformation"],
                    "intensity": annotation["Intensity"],
                    "phase": annotation["Phase"],
                    "insar_path": insar_path,
                    "mask_path": mask_path,
                    }

                if 0 not in record["label"]:
                    self.positives.append(record)
                else:
                    self.negatives.append(record)

                self.records.append(record)
                self.frameIDs.append(record["frameID"])

            except FileNotFoundError as e:
                self.missing_file_list.append(file)
                self.missing_file_count += 1
                continue

        self.num_examples = len(self.records)
        self.frameIDs = np.unique(self.frameIDs)
        self.frame_dict = {}
        for idx, frame in enumerate(self.frameIDs):
            self.frame_dict[frame] = idx

        print(mode + " Number of ground deformation samples: ", len(self.positives))
        print(mode + " Number of non deformation samples: ", len(self.negatives))
        print(mode + " Number of missing samples: ", self.missing_file_count)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor]:
        label = None
        if self.oversampling and self.mode == "train":
            choice = random.randint(0, 1)
            if choice == 0:
                choice_neg = random.randint(0, len(self.negatives) - 1)
                sample = self.negatives[choice_neg]
            else:
                choice_pos = random.randint(0, len(self.positives) - 1)
                sample = self.positives[choice_pos]
        else:
            sample = self.records[idx]

        insar = ImageFile.Image.open(sample["insar_path"]).convert('RGB')
        
        if self.transform:
            insar = self.transform(insar)
        else:
            insar = insar
        
        if 0 not in sample["label"]:
            label = 1.0
            label = torch.tensor(label).long()
        else:
            label = 0.0
            label = torch .tensor(label).long()

        return insar, label

class CustomImageFolderFT(Dataset):
    def __init__(self, dir: str, transform=None) -> None:
        self.paths = list(Path(dir).glob("*/*.png"))
        self.transform = transform

    def load_image(self, idx: int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path).convert("L").convert("RGB") # grayscale then 3 channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(idx)
        class_label = self.paths[idx].parent.name

        if self.transform:
            return self.transform(img), class_label
        else:
            return img, int(class_label)

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)



        



