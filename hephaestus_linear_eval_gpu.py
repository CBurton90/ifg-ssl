import sys
sys.path.append("/home/conradb/git/ifg-ssl")
from collections import defaultdict, Counter, OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# local module imports
import dino.vision_transformer as vit
from dino.linear_classifier import LinearClassifier, train, validate

model = 'vit_small' # embed_dim=384 depth=12, num_heads=6
n_last_blocks = 4
avgpool_patchtokens = False
num_labels = 2
batch_size= 256
num_workers= 32
pin_memory= True
epochs = 101
lr = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Multi2UniLabelTfm():
#     def __init__(self,pos_label=5):
#         if isinstance(pos_label,int) or isinstance(pos_label,float):
#             pos_label = [pos_label,]
#         self.pos_label = pos_label

#     def __call__(self,y):
#         # if y==self.pos_label:
#         if y in self.pos_label:
#             return 1
#         else:
#             return 0

model = vit.__dict__[model](patch_size=16, num_classes=0)
embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

state_dict = torch.load('dino/dino_checkpoints/t60pct_hephaestus_ViTS16_ckpt_v1_final.pth', map_location='cpu')
state_dict = state_dict["teacher"]

# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

linear_classifier = LinearClassifier(embed_dim, num_labels=num_labels)

train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

val_transform = transforms.Compose([
        # transforms.Resize(256, interpolation=3),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

# target_transform = Multi2UniLabelTfm(pos_label=[1,2,3,4,5,6])
dataset = datasets.ImageFolder(root='/scratch/SDF25/Hephaestus_Classification/InSARLabelledBinary', transform=None)

# Stratified Sampling for train and val
train_idx, validation_idx = train_test_split(np.arange(len(dataset)),
                                             test_size=0.4,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=dataset.targets)



# print(dataset.targets)
# print([dataset.targets[i] for i in validation_idx])

# Subset dataset for train and val
train_dataset = Subset(dataset, train_idx)
validation_dataset = Subset(dataset, validation_idx)

val2idx, testidx = train_test_split(np.arange(len(validation_dataset)),
                                             test_size=0.5,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=[dataset.targets[i] for i in validation_idx])

validation_dataset2 = Subset(validation_dataset, val2idx)
val_targets = [dataset.targets[i] for i in validation_idx]
val_targets2 = [val_targets[i] for i in val2idx]
val_c = Counter(val_targets2)
val_c = OrderedDict(sorted(val_c.items()))
print(f'Val n samples is {len(validation_dataset2)}')
print(f'Val unbalanced sampling class distribution {val_c}')
# val2, test_dataset = torch.utils.data.random_split(validation_dataset, [0.5, 0.5])
# val_targets = [i for i in val2.targets]
# print(val_targets)


# Weighted sampling to address class imbalance
train_targets = [dataset.targets[i] for i in train_idx]
train_c = Counter(train_targets)
train_c = OrderedDict(sorted(train_c.items()))
print(f'Train balanced sampling class distribution {train_c}')
train_length = 0
for value in train_c.values():
    train_length += int(value)
assert train_length == len(train_dataset)

# calcuate weight of each class
train_class_weights = [1.0/n for n in train_c.values()]
print(f'Train class weights {train_class_weights}')

# assign weight to each sample
train_sample_weights = [train_class_weights[n] for n in train_targets]

# Create WeightedRandomSampler
sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))

def collate_fn(batch):
    data = [train_transform(item[0]) for item in batch]
    target = [torch.tensor(item[1]).to(torch.long) for item in batch]
    return (torch.stack(data), torch.stack(target))

# assign sampler
train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

def val_collate_fn(batch):
    data = [val_transform(item[0]) for item in batch]
    target = [torch.tensor(item[1]).to(torch.long) for item in batch]
    return (torch.stack(data), torch.stack(target))

val_dataloader = DataLoader(validation_dataset2, shuffle=False, batch_size=batch_size, collate_fn=val_collate_fn, num_workers=num_workers, pin_memory=pin_memory)

# set optimizer
optimizer = torch.optim.SGD(linear_classifier.parameters(),
                            lr,
                            momentum=0.9,
                            weight_decay=0.0001, # we do not apply weight decay
                            )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

for epoch in range(0, epochs):
    total_loss, t_counts = train(model, linear_classifier, optimizer, train_dataloader, epoch, n_last_blocks, avgpool_patchtokens, device)
    val_acc, val_loss, v_counts = validate(val_dataloader, model, linear_classifier, n_last_blocks, avgpool_patchtokens, device)
    scheduler.step()

    epoch_loss = total_loss / t_counts
    val_loss = val_loss / v_counts
    acc = val_acc / v_counts
    print(f'Epoch {epoch} of {epochs}')
    print(f'train loss is {epoch_loss}')
    print(f'val loss is {val_loss}')
    print(f'validation accuracy is {acc}')

# for i, (a, b) in enumerate(train_dataloader):
#     # print(a.shape)
#     # print(b.shape)
#     if i == 10:
#         plt.imsave('hephaestus_attn_plots/example_itr_'+str(i)+'batch_ex_0_class_'+str(b[0].item())+'.png',
#                    a[0, :, :, :].cpu().numpy().transpose(1,2,0))

