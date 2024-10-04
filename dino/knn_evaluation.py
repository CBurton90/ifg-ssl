import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import torch

import dino.utils as utils

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    
    features = None
    
    for idx, (samples, label) in enumerate(data_loader):
        
        samples = samples.cuda(non_blocking=True)
        index = idx.cuda(non_blocking=True)
        
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
        if use_cuda:
            features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            
        features = torch.cat((features, feats),0)

    return features

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=2):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 2
    print(num_test_images, num_chunks)
    imgs_per_chunk = num_test_images // num_chunks
    print(imgs_per_chunk)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    print(retrieval_one_hot)
    for idx in range(0, num_test_images, imgs_per_chunk):
        print(idx)
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        print(features.shape)
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        print(targets.shape)
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        print(similarity.shape)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        print(distances.shape, indices.shape)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        print(train_labels.view(1, -1).expand(batch_size, -1).shape)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        print(f'top 20 closest train labels per batch {retrieved_neighbors.shape}')

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        print(retrieval_one_hot.shape)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        print(retrieval_one_hot.shape)
        distances_transform = distances.clone().div_(T).exp_()
        print(distances.clone().div_(T))
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), distances_transform.view(batch_size, -1, 1),),1,)
        print(probs)
        _, predictions = probs.sort(1, True)
        print(f'predictions {predictions}')

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        print(targets.data.view(-1, 1))
        print(correct)
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        print(correct.narrow(1, 0, 1))
        # top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    print(top1)
    # top5 = top5 * 100.0 / total
    return top1#, top5

if __name__ == '__main__':
    train_feat = (torch.randint(-100, 100, (2000,192)) * 1e-5).float()
    train_lab = torch.randint(0, 2, (2000,1)).long()
    test_feat = (torch.randint(-100, 100, (32,192)) * 1e-5).float()
    test_lab = torch.randint(0, 2, (32,1)).long()
    print(test_lab)

    a = knn_classifier(train_feat, train_lab, test_feat, test_lab, 20, 0.07, num_classes=2)