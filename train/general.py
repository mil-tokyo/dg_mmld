from torch import nn
from util.util import split_domain
import torch
from numpy.random import *
import numpy as np
from loss.EntropyLoss import HLoss
from loss.MaximumSquareLoss import MaximumSquareLoss

def train(model, train_data, optimizers, device, epoch, num_epoch, filename, entropy, disc_weight=None, entropy_weight=1.0, grl_weight=1.0):
    class_criterion = nn.CrossEntropyLoss()
    print(disc_weight)
    domain_criterion = nn.CrossEntropyLoss(weight=disc_weight)
    if entropy == 'default':
        entropy_criterion = HLoss()
    else:
        entropy_criterion = MaximumSquareLoss()
    p = epoch / num_epoch
    alpha = (2. / (1. + np.exp(-10 * p)) -1) * grl_weight
    beta = (2. / (1. + np.exp(-10 * p)) -1) * entropy_weight
    model.discriminator.set_lambd(alpha)
    model.train()  # Set model to training mode
    running_loss_class = 0.0
    running_correct_class = 0
    running_loss_domain = 0.0
    running_correct_domain = 0
    running_loss_entropy = 0
    # Iterate over data.
    for inputs, labels, domains in train_data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        # zero the parameter gradients
        for optimizer in optimizers:
            optimizer.zero_grad()
        # forward
        output_class, output_domain = model(inputs)

        loss_class = class_criterion(output_class, labels)
        loss_domain = domain_criterion(output_domain, domains)
        loss_entropy = entropy_criterion(output_class)
        _, pred_class = torch.max(output_class, 1)
        _, pred_domain = torch.max(output_domain, 1)

        total_loss = loss_class + loss_domain + loss_entropy * beta
        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        running_loss_class += loss_class.item() * inputs.size(0)
        running_correct_class += torch.sum(pred_class == labels.data)
        running_loss_domain += loss_domain.item() * inputs.size(0)
        running_correct_domain += torch.sum(pred_domain == domains.data)
        running_loss_entropy += loss_entropy.item() * inputs.size(0)

    epoch_loss_class = running_loss_class / len(train_data.dataset)
    epoch_acc_class = running_correct_class.double() / len(train_data.dataset)
    epoch_loss_domain = running_loss_domain / len(train_data.dataset)
    epoch_acc_domain = running_correct_domain.double() / len(train_data.dataset)
    epoch_loss_entropy = running_loss_entropy / len(train_data.dataset)
    
    log = 'Train: Epoch: {} Alpha: {:.4f} Loss Class: {:.4f} Acc Class: {:.4f}, Loss Domain: {:.4f} Acc Domain: {:.4f} Loss Entropy: {:.4f}'.format(epoch, alpha, epoch_loss_class, epoch_acc_class, epoch_loss_domain, epoch_acc_domain, epoch_loss_entropy)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n') 
    return model, optimizers