import sys
sys.path.append("/home/conradb/git/ifg-ssl")
from S1_C1.configs.config import load_global_config
from dino.knn_evaluation import extract_features, knn_classifier
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dino.vision_transformer as vit
import torch

def extract_feature_pipeline(config):
    
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
        ])
        
    train_dataset = datasets.ImageFolder(root=config.data.train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=config.data.val_path, transform=transform)

    print(f'val data is\n {val_dataset[0][0].shape}\n{val_dataset[0][0]}')

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.train_eval_batch_size,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ============ building network ... ============
    eval_model = vit.__dict__[config.model.model](patch_size=config.model.patch_size)
    eval_model.to(config.train.device)
    
    state_dict = torch.load('../'+config.model.checkpoint_path, map_location="cpu")
    state_dict = state_dict['teacher']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = eval_model.load_state_dict(state_dict, strict=False)
    
    eval_model.eval()

    print("Extracting features for train set...")
    train_features, train_labels = extract_features(eval_model, data_loader_train, use_cuda=True)
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(eval_model, data_loader_val, use_cuda=True)

    #if utils.get_rank() == 0:
    train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)

    if config.data.c1_val:

        transform2 = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
            ])
        
        c1_val_dataset = datasets.ImageFolder(root=config.data.c1_val_path, transform=transform2)

        print(f'C1 data is\n {c1_val_dataset[0][0].shape}\n{c1_val_dataset[0][0]}')

        c1_data_loader_val = torch.utils.data.DataLoader(
            c1_val_dataset,
            batch_size=config.train.c1_val_batch_size,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            )
        
        print("Extracting features for C1 val set...")
        c1_features, c1_labels = extract_features(eval_model, c1_data_loader_val, use_cuda=True)
        c1_features = torch.nn.functional.normalize(c1_features, dim=1, p=2)
        
        return train_features, train_labels, test_features, test_labels, c1_features, c1_labels
    
    else:
        return train_features, train_labels, test_features, test_labels

if __name__ == '__main__':
    config = load_global_config('../configs/dino_S1_train.toml')
    if config.data.c1_val:
        train_features, train_labels, test_features, test_labels, c1_features, c1_labels = extract_feature_pipeline(config)
        top1_s1 = knn_classifier(train_features, train_labels, test_features, test_labels, 10, 0.07, num_classes=2)
        top1_c1 = knn_classifier(train_features, train_labels, c1_features, c1_labels, 20, 0.07, num_classes=2)
        print(f'Top1 accuracy for S1 validation set is {top1_s1}')
        print(f'Top1 accuracy for C1 validation set is {top1_c1}')
    else:
        train_features, train_labels, test_features, test_labels = extract_feature_pipeline(config)
        top1_s1 = knn_classifier(train_features, train_labels, test_features, test_labels, 2, 0.07, num_classes=2)
        print(f'Top1 accuracy for S1 validation set is {top1_s1}')

