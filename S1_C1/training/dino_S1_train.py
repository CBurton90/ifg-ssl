# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import math
from collections import Counter, OrderedDict
from tqdm import tqdm

# torch imports
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# local module imports
from S1_C1.configs.config import load_global_config
from S1_C1.utils.utils import calculate_sampler_weights
from dino.augment import IfgAugmentationDINO
from dino.knn_evaluation import extract_features, knn_classifier
from dino.loss import DINOLoss
import dino.utils as utils
import dino.vision_transformer as vit
from dino.vision_transformer import DINOHead


def train_dino(config):
    device = torch.device(config.train.device)
    print(f'Using {device}')

    transform = IfgAugmentationDINO(tuple(config.crops.global_crops_scale),
                                tuple(config.crops.local_crops_scale),
                                config.crops.local_crops_number,
                                tuple(config.data.mean),
                                tuple(config.data.std))
    
    dataset = datasets.ImageFolder(root=config.data.train_path, transform=transform)

    sample_weights = calculate_sampler_weights(dataset)

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataloader = DataLoader(dataset,
                              sampler=sampler,
                              batch_size=config.train.batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    student = vit.__dict__[config.model.model](patch_size=config.model.patch_size, drop_path_rate=0.1)
    teacher = vit.__dict__[config.model.model](patch_size=config.model.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(embed_dim, config.model.out_dim, use_bn=config.model.use_bn_head, norm_last_layer=config.model.norm_last_layer))
    teacher = utils.MultiCropWrapper(teacher, DINOHead(embed_dim, config.model.out_dim, use_bn=config.model.use_bn_head))

    student, teacher = student.to(device), teacher.to(device)
    
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(
        config.model.out_dim,
        config.crops.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        config.DINO.warmup_teacher_temp,
        config.DINO.teacher_temp,
        config.DINO.warmup_teacher_temp_epochs,
        config.train.epochs,
        ).to(device)

    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    lr_schedule = utils.cosine_scheduler(config.train.lr, config.train.min_lr, config.train.epochs, len(train_dataloader), warmup_epochs=config.train.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(config.train.weight_decay, config.train.weight_decay_end, config.train.epochs, len(train_dataloader))

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(config.DINO.momentum_teacher, 1, config.train.epochs, len(train_dataloader))
    
    fp16_scaler = None
    if config.train.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    print('Commencing training')

    for epoch in range(0, config.train.epochs):

        print(f'Epoch {epoch}/{config.train.epochs}')

        epoch_loss = train_one_epoch(student, teacher, dino_loss, train_dataloader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, config)
        print(f'Epoch loss is {epoch_loss}')

        if epoch % 5 == 0:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'dino_loss': dino_loss.state_dict(),
                }
            print('saving checkpoint')
            torch.save(save_dict, config.model.checkpoint_path)

            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(config)
            top1 = knn_classifier(train_features, train_labels, test_features, test_labels, 10, 0.07, num_classes=2) #10 NN and 0.07 temp
            print(f'Top1 accuracy for validation set is {top1} size is {test_labels.shape}')


        

def train_one_epoch(student, teacher, dino_loss, train_dataloader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, config):

    running_loss = 0
    counts = 0

    for it, (batch, label) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):

        it = len(train_dataloader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        # move images to gpu
        images = [image.to(torch.device(config.train.device)) for image in batch]

        #print(images[0].shape)
        #print(images[3].shape)
        #print(label.shape)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        running_loss += loss.detach().cpu().numpy()
        counts += 1

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None

        # if not using fp16
        if fp16_scaler is None:
            loss.backward()
            if config.train.clip_grad:
                param_norms = utils.clip_gradients(student, config.train.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, config.train.freeze_last_layer)
            optimizer.step()
        # if using fp16
        else:
            fp16_scaler.scale(loss).backward()
            if config.train.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, config.train.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, config.train.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    epoch_loss = running_loss / counts

    return epoch_loss



def extract_feature_pipeline(config):
    
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(tuple(config.data.mean), tuple(config.data.std)),
    ])
    train_dataset = datasets.ImageFolder(root=config.data.train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=config.data.val_path, transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
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
    
    state_dict = torch.load(config.model.checkpoint_path, map_location="cpu")
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

    
    #train_labels = torch.tensor([s[-1] for s in train_dataset.samples]).long()
    #test_labels = torch.tensor([s[-1] for s in val_dataset.samples]).long()
    # save features and labels
    # if args.dump_features and dist.get_rank() == 0:
    #     torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
    #     torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
    #     torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
    #     torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


if __name__ == '__main__':
    config = load_global_config('configs/dino_S1_train.toml')
    train_dino(config)
