import torch
import torch.nn as nn

from dino.utils import accuracy

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, device):
    linear_classifier.train()
    linear_classifier.to(device)
    model.to(device)

    running_loss = 0
    counts = 0

    for it, (inp, target) in enumerate(loader):
        # move to gpu
        inp = inp.to(device)
        target = target.to(device)

        # forward for ViT
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
                
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)
        running_loss += loss
        counts += 1

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # print(f'loss is {loss}')

        # step
        optimizer.step()

    return running_loss, counts

@torch.no_grad()
def validate(val_loader, model, linear_classifier, n, avgpool, device):
    linear_classifier.eval()
    linear_classifier.to(device)
    model.to(device)

    running_loss = 0
    running_acc = 0
    counts = 0

    for it, (inp, target) in enumerate(val_loader):
        # move to gpu
        inp = inp.to(device)
        target = target.to(device)

        # forward for ViT
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)

        output = linear_classifier(output)

        loss = nn.CrossEntropyLoss()(output, target)
        running_loss += loss
        counts += 1

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        running_acc += acc1

    return running_acc, running_loss, counts