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
from dino.linear_classifier import LinearClassifier, train

model = 'vit_tiny' # embed_dim=192, depth=12, num_heads=3
n_last_blocks = 4
avgpool_patchtokens = False
num_labels = 10
batch_size= 256
num_workers= 32
pin_memory= True
epochs = 101
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vit.__dict__[model](patch_size=16, num_classes=0)
embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

state_dict = torch.load('dino/dino_checkpoints/dino_imagenette_160_ckpt.pth', map_location='cpu')
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

path = untar_data(URLs.IMAGENETTE_160)
train_dataset = datasets.ImageFolder(root=path/'train', transform=train_transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

# set optimizer
optimizer = torch.optim.SGD(linear_classifier.parameters(),
                            lr,
                            momentum=0.9,
                            weight_decay=0, # we do not apply weight decay
                            )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

for epoch in range(0, epochs+1):

        total_loss, counts = train(model, linear_classifier, optimizer, train_loader, epoch, n_last_blocks, avgpool_patchtokens, device)
        scheduler.step()

        epoch_loss = total_loss / counts
        print(f'Epoch {epoch} loss is {epoch_loss}')
