# standard python imports
import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# pytorch imports
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional.classification import binary_recall, binary_precision
from timm.models.layers import trunc_normal_

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from utils.config import load_global_config
from hephaestus.dataset.Dataset import FullFrameDataset, CustomImageFolderFT, DatasetFromSubset
import MAE.models_vit as models_vit
from MAE.pos_embed import interpolate_pos_embed
from MAE.utils import LARS, NativeScalerWithGradNormCount as NativeScaler, RandomResizedCrop, param_groups_lrd
from MAE.engine_finetune import train_one_epoch, evaluate

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set:
    torch.backends.cudnn.deterministic = True # Use deterministic algorithms.
    torch.backends.cudnn.benchmark = False # Causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(config):

    # Set random seed and device
    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed()
    device = torch.device(config.train.device)

    train_transform = transforms.Compose([
        # transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.ifg_mean, std=config.data.ifg_std),
        ])

    val_transform = transforms.Compose([
        # transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        # transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.ifg_mean, std=config.data.ifg_std),
        ])

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3), # Grayscale transform, TODO check if MAE will function with single channel images
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.ifg_mean, std=config.data.ifg_std),
        ])

    # train_dataset = FullFrameDataset(config, mode="train", transform=train_transform)
    # val_dataset = FullFrameDataset(config, mode="val", transform=val_transform)
    # test_dataset = FullFrameDataset(config, mode="test", transform=test_transform)

    ds = CustomImageFolderFT(dir='/scratch/SDF25/InSAR-ssl-finetune/LiCS-NZ-syn/', transform=None)
    train_ss, val_ss = torch.utils.data.random_split(ds, [0.8, 0.2])
    train_ds = DatasetFromSubset(train_ss, transform=train_transform)
    val_ds = DatasetFromSubset(val_ss, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=False,
        )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=config.train.test_batch_size,
    #     shuffle=False,
    #     num_workers=config.train.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    #     )

    model = models_vit.__dict__[config.model.model](
        num_classes=config.data.num_classes,
        drop_path_rate=config.model.drop_path,
        global_pool=config.model.global_pool,
    )

    checkpoint = torch.load(config.model.checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % config.model.checkpoint_path)
    checkpoint_model = checkpoint['model']
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

        val_acc, val_loss, v_counts, _, _ = evaluate(val_loader, model, device)
        # test_acc, test_loss, test_counts, probs_list, labels_list = evaluate(test_loader, model, device)

        # print(probs_list[-5:])
        # print(labels_list[-5:])
        # print(probs_list.shape)
        # print(labels_list.shape)

        val_loss = val_loss / v_counts
        # test_loss = test_loss / test_counts
        acc = val_acc / v_counts
        # test_acc = test_acc / test_counts
        print(f'Epoch {epoch} of {config.train.epochs}')
        print(f'train loss is {train_loss}')
        print(f'val loss is {val_loss}')
        # print(f'test loss is {test_loss}')
        print(f'validation accuracy is {acc}')
        # print(f'test accuracy is {test_acc}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)
        # test_epoch_loss.append(test_loss) 
        val_epoch_acc.append(acc)
        # test_epoch_acc.append(test_acc)

        # recall.append(binary_recall(probs_list, labels_list, threshold=0.5).cpu())
        # precision.append(binary_precision(probs_list, labels_list, threshold=0.5).cpu())


        if epoch % 5 == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            fig.suptitle('Hephaestus MAE Fine-tuning (Train, Val, Test)')
            ax1.plot(range(len(train_epoch_loss)), np.array(train_epoch_loss), 'b', label='train')
            ax1.plot(range(len(val_epoch_loss)), np.array(val_epoch_loss), 'c', label='val')
            # ax1.plot(range(len(test_epoch_loss)), np.array(test_epoch_loss), 'm', label='test')
            ax1.legend(loc="upper right")
            ax1.set_ylabel('Cross Entropy Loss')

            ax2.plot(range(len(val_epoch_acc)), np.array(val_epoch_acc), 'c', label='val')
            # ax2.plot(range(len(test_epoch_acc)), np.array(test_epoch_acc), 'm', label='test')
            ax2.legend(loc="upper right")
            ax2.set_ylabel('Accuracy')

            # ax3.plot(range(len(recall)), np.array(recall), 'c', label = 'test recall (tp / (tp + fn))')
            # ax3.plot(range(len(precision)), np.array(precision), 'm', label = 'test precision (tp / (tp + fp))')
            # ax3.legend(loc="upper right")
            # ax3.set_ylabel('Ratio (%)')
            plt.xlabel("Epochs")
            plt.savefig('tmp/MAE_end-to-end_finetuning_'+str(config.train.epochs)+'_epochs.png', dpi=300, format='png')

            # fig, ax4 = plt.subplots(1)
            # fig.suptitle('Hephaestus MAE Fine-tuning ROC Curve (Test)')
            # # calculate roc curves
            # ns_probs = [0 for _ in range(len(labels_list))]
            # ns_fpr, ns_tpr, _ = roc_curve(np.array(labels_list.cpu()), ns_probs)
            # lr_fpr, lr_tpr, _ = roc_curve(np.array(labels_list.cpu()), np.array(probs_list.detach().cpu()))

            # # plot the roc curve for the model
            # ax4.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            # ax4.plot(lr_fpr, lr_tpr, marker='.', label='MAE Fine-tuned')
            # ax4.legend(loc="lower right")
            # # axis labels
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.savefig('tmp/MAE_ROC_curve_finetuning_'+str(config.train.epochs)+'_epochs.png', dpi=300, format='png')

            save_dict = {
                'epoch': epoch,
                'arch' : config.model.model,
                'model': model.state_dict(),
                }
            print('saving checkpoint')
            torch.save(save_dict, config.model.checkpoint_save_path)


if __name__ == '__main__':
    config = load_global_config('configs/MAE_finetuning_hephaestus.toml')
    main(config)