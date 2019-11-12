from torch import nn
from util.util import split_domain
import torch
from numpy.random import *
import numpy as np

def train(model, train_data, optimizers, device, epoch, num_epoch, filename, entropy=None, disc_weight=None, entropy_weight=None, grl_weight=None):
    criterion = nn.CrossEntropyLoss()

    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in train_data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        for optimizer in optimizers:
            optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_data.dataset)
    epoch_acc = running_corrects.double() / len(train_data.dataset)
    
    log = 'Train: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n') 
    return model, optimizers
