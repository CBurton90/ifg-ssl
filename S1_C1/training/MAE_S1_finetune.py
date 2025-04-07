# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import random
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.models.layers import trunc_normal_
from torcheval.metrics.functional.classification import binary_recall, binary_precision

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import MAE.models_vit as models_vit
from MAE.pos_embed import interpolate_pos_embed
from MAE.utils import LARS, NativeScalerWithGradNormCount as NativeScaler, RandomResizedCrop, param_groups_lrd
from MAE.engine_finetune import train_one_epoch, evaluate
from S1_C1.configs.config import load_global_config
from S1_C1.utils.utils import calculate_sampler_weights

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(config):
    
    set_seed()
    device = torch.device(config.train.device)

    transform_train = transforms.Compose([
            # RandomResizedCrop(224, interpolation=3),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std)])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std)])
    transform_c1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std),
            ])
    
    dataset_train = datasets.ImageFolder(config.data.train_path, transform=transform_train)
    dataset_val = datasets.ImageFolder(config.data.val_path, transform=transform_val)
    dataset_c1 = datasets.ImageFolder(config.data.c1_path, transform=transform_c1)

    # Oversampling
    sample_weights = calculate_sampler_weights(dataset_train)
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(dataset_train,
                              sampler=sampler,
                              batch_size=config.train.batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    val_loader = DataLoader(dataset_val,
                              batch_size=config.train.val_batch_size,
                              num_workers=config.train.num_workers,
                              pin_memory=config.train.pin_memory,
                              drop_last=True)
    
    test_loader = DataLoader(dataset_c1,
                               batch_size=config.train.c1_batch_size,
                               num_workers=config.train.num_workers,
                               pin_memory=config.train.pin_memory,
                               drop_last=False)

    model = models_vit.__dict__[config.model.model](
        num_classes=config.data.num_classes,
        drop_path_rate=config.model.drop_path,
        global_pool=config.model.global_pool,
        )

    checkpoint = torch.load(config.model.checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % config.model.checkpoint_path)
    checkpoint_model = checkpoint['MAE_encoder']
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if config.model.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

     # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    config.train.lr = config.train.blr * config.train.batch_size / 256
    print("base lr: %.2e" % (config.train.lr * 256 / config.train.batch_size))
    print("actual lr: %.2e" % config.train.lr)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = param_groups_lrd(model, config.train.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=config.model.layer_decay
    )

    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Start training for {config.train.epochs} epochs")

    train_epoch_loss = []
    val_epoch_loss = []
    test_epoch_loss = []
    val_epoch_acc = []
    test_epoch_acc = []
    recall = []
    precision = []

    for epoch in range(0, config.train.epochs):
        
        train_loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=None,
            config=config)

        val_acc, val_loss, v_counts, probs_list, labels_list = evaluate(val_loader, model, device)
        test_acc, test_loss, test_counts, _, _ = evaluate(test_loader, model, device)

        print(probs_list[-5:])
        print(labels_list[-5:])
        print(probs_list.shape)
        print(labels_list.shape)

        val_loss = val_loss / v_counts
        test_loss = test_loss / test_counts
        acc = val_acc / v_counts
        test_acc = test_acc / test_counts
        print(f'Epoch {epoch} of {config.train.epochs}')
        print(f'train loss is {train_loss}')
        print(f'val loss is {val_loss}')
        print(f'test loss is {test_loss}')
        print(f'validation accuracy is {acc}')
        print(f'test accuracy is {test_acc}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)
        test_epoch_loss.append(test_loss) 
        val_epoch_acc.append(acc)
        test_epoch_acc.append(test_acc)

        recall.append(binary_recall(probs_list, labels_list, threshold=0.5).cpu())
        precision.append(binary_precision(probs_list, labels_list, threshold=0.5).cpu())


        if epoch % 5 == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            fig.suptitle('Synthetic InSAR MAE Fine-tuning (Train, Val)')
            ax1.plot(range(len(train_epoch_loss)), np.array(train_epoch_loss), 'b', label='train')
            ax1.plot(range(len(val_epoch_loss)), np.array(val_epoch_loss), 'c', label='val')
            # ax1.plot(range(len(test_epoch_loss)), np.array(test_epoch_loss), 'm', label='test')
            ax1.legend(loc="upper right")
            ax1.set_ylabel('Cross Entropy Loss')

            ax2.plot(range(len(val_epoch_acc)), np.array(val_epoch_acc), 'c', label='val')
            # ax2.plot(range(len(test_epoch_acc)), np.array(test_epoch_acc), 'm', label='test')
            ax2.legend(loc="upper right")
            ax2.set_ylabel('Accuracy')

            ax3.plot(range(len(recall)), np.array(recall), 'c', label = 'val recall (tp / (tp + fn))')
            ax3.plot(range(len(precision)), np.array(precision), 'm', label = 'val precision (tp / (tp + fp))')
            ax3.legend(loc="upper right")
            ax3.set_ylabel('Ratio')
            plt.xlabel("Epochs")
            plt.savefig('outputs/syn_MAE_end-to-end_finetuning_'+str(config.train.epochs)+'_epochs.png', dpi=300, format='png')

            fig, ax4 = plt.subplots(1)
            fig.suptitle('Synthetic InSAR MAE Fine-tuning ROC Curve (Val)')
            # calculate roc curves
            ns_probs = [0 for _ in range(len(labels_list))]
            ns_fpr, ns_tpr, _ = roc_curve(np.array(labels_list.cpu()), ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(np.array(labels_list.cpu()), np.array(probs_list.detach().cpu()))

            # plot the roc curve for the model
            ax4.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            ax4.plot(lr_fpr, lr_tpr, marker='.', label='MAE Fine-tuned')
            ax4.legend(loc="lower right")
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.savefig('outputs/syn_MAE_ROC_curve_finetuning_'+str(config.train.epochs)+'_epochs.png', dpi=300, format='png')

if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_global_config('configs/MAE_S1_finetuning.toml')
    main(config)