# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")

# pytorch + fastai imports
from fastai.vision.all import *
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# local module imports
import dino.vision_transformer as vit
from dino.linear_classifier import LinearClassifier, train, validate

model = 'vit_small' # embed_dim=384, depth=12, num_heads=6
n_last_blocks = 4
avgpool_patchtokens = False
num_labels = 10
batch_size= 256
num_workers= 32
pin_memory= True
epochs = 201
lr = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vit.__dict__[model](patch_size=16, num_classes=0)
embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

state_dict = torch.load('dino/dino_checkpoints/dino_imagenette_320_ckpt_v1.pth', map_location='cpu')
state_dict = state_dict["teacher"]

# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

linear_classifier = LinearClassifier(embed_dim, num_labels=num_labels)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

path = untar_data(URLs.IMAGENETTE_320)
train_dataset = datasets.ImageFolder(root=path/'train', transform=train_transform)
val_dataset = datasets.ImageFolder(root=path/'val', transform=val_transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

# set optimizer
optimizer = torch.optim.SGD(linear_classifier.parameters(),
                            lr,
                            momentum=0.9,
                            weight_decay=0, # we do not apply weight decay
                            )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

for epoch in range(0, epochs):

        total_loss, t_counts = train(model, linear_classifier, optimizer, train_loader, epoch, n_last_blocks, avgpool_patchtokens, device)
        val_acc, val_loss, v_counts = validate(val_loader, model, linear_classifier, n_last_blocks, avgpool_patchtokens, device)
        scheduler.step()

        epoch_loss = total_loss / t_counts
        val_loss = val_loss / v_counts
        acc = val_acc / v_counts
        print(f'Epoch {epoch} of {epochs}')
        print(f'train loss is {epoch_loss}')
        print(f'val loss is {val_loss}')
        print(f'validation accuracy is {acc}')
