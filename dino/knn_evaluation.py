import sys
sys.path.append("/home/conradb/git/ifg-ssl")
import numpy as np
import torch

import dino.utils as utils

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    
    features = None
    features2 = []
    labels = []
    
    for idx, (samples, label) in enumerate(data_loader):
        
        samples = samples.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        #index = idx.cuda(non_blocking=True)
        
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
            #print(feats.shape)

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
        if use_cuda:
            features = features.cuda(non_blocking=True)
            #print(f"Storing features into tensor of shape {features.shape}")
            
        #features = torch.cat((features, feats),0)
        features2.append(feats)
        labels.append(label)
    output = torch.stack(features2).reshape(-1, feats.shape[-1])
    labels = torch.stack(labels).reshape(-1, 1).long()
    labels = torch.tensor(labels).long()
    print(f'feature output is dim {output.shape}')
    print(f'label output is dim {labels.shape}')

    return output, labels

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=2):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    print(f'TRAIN FEATURES ARE SIZE\n {train_features.shape}')
    num_test_images, num_chunks = test_labels.shape[0], 1
    #print(num_test_images, num_chunks)
    imgs_per_chunk = num_test_images // num_chunks
    #print(imgs_per_chunk)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    #print(retrieval_one_hot)
    for idx in range(0, num_test_images, imgs_per_chunk):
        #print(idx)
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        print(f'TEST FEATURES ARE\n {features.shape}\n{features}')
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        print(f'TEST LABELS ARE\n {targets.shape}\n{targets}')
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        print(f'DOT PROD OF TEST FEATURES x TRAIN FEATURES\n {similarity.shape}')
        #print(similarity.shape)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        print(f'TOP K DOT PROD VALS ARE\n {distances.shape}\n{distances}')
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        print(f'TRAIN LABEL CANDIDATES ARE\n {candidates.shape}')
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        print(f'RETRIEVED CANDIDATES ARE\n {retrieved_neighbors.shape}\n{retrieved_neighbors}')

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        #print(retrieval_one_hot.shape)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        print(f'ONE HOT RETRIEVED CANDIATES ARE\n {retrieval_one_hot.shape}\n{retrieval_one_hot}')
        distances_transform = distances.clone().div_(T).exp_()
        print(f'EXP(TOP K VALS / TEMP)\n {distances_transform.shape}\n{distances_transform}')
        onehotXvals = torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), distances_transform.view(batch_size, -1, 1),)
        print(f'ONE HOT x EXP(TOP K VALS)\n {onehotXvals.shape}\n{onehotXvals}')
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), distances_transform.view(batch_size, -1, 1),),1,)
        print(f'PROBS ARE\n {probs.shape}\n{probs}')
        _, predictions = probs.sort(1, True)
        print(f'PREDICTION IDX/CLASS SORTED\n{predictions}')

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        print(f'CORRECT PREDS\n{correct.narrow(1, 0, 1)}')
        # top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    print(f'Top 1 acc {top1}')
    # top5 = top5 * 100.0 / total
    return top1#, top5

#if __name__ == '__main__':
    #train_feat = (torch.randint(-5, 5, (2000,192)) * 1e-5).float()
    #train_lab = torch.randint(0, 2, (2000,1)).long()
    #test_feat = (torch.randint(-5, 5, (32,192)) * 1e-5).float()
    #test_lab = torch.randint(0, 2, (32,1)).long()

    #a = knn_classifier(train_feat, train_lab, test_feat, test_lab, 20, 0.07, num_classes=2)
