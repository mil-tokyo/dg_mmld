from model import alexnet, caffenet
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from torch.nn import init
from model import caffenet, alexnet, resnet
from dataloader.dataloader import *
from clustering.domain_split import calc_mean_std
from sklearn.decomposition import PCA
from util.scheduler import inv_lr_scheduler


def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        print('model.features parameters are fixed')
        for param in model.parameters():
            param.requires_grad = False
            
def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print('Source domain: ', end='')
        for domain in source_domain:
            print(domain, end=', ')
        print('Target domain: ', end='')
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain
    
domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ['Caltech', 'Labelme', 'Pascal', 'Sun']
}

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

nets_map = {
    'caffenet': {'deepall': caffenet.caffenet, 'general': caffenet.DGcaffenet},
    'alexnet': {'deepall': alexnet.alexnet, 'general': alexnet.DGalexnet},
    'resnet': {'deepall': resnet.resnet, 'general': resnet.DGresnet}
}

def get_model(name, train):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name][train](**kwargs)

    return get_network_fn

def get_model_lr(name, train, model, fc_weight=1.0, disc_weight=1.0):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)
    if train not in train_map:
        raise ValueError('Name of train unknown %s' % name)
        
    if name == 'alexnet' and train == 'deepall':
        return  [(model.features, 1.0), (model.classifier[:-1], 1.0), (model.classifier[-1], 1.0* fc_weight)]
    elif name == 'caffenet' and train == 'deepall':
        return [(model.features, 1.0), (model.classifier, 1.0), (model.class_classifier, 1.0 * fc_weight)]
    elif name == 'resnet' and train == 'deepall':
        return [(model.conv1, 1.0), (model.bn1, 1.0), (model.layer1, 1.0), (model.layer2, 1.0), (model.layer3, 1.0), 
               (model.layer4, 1.0), (model.fc, 1.0 * fc_weight)]
    
    elif name == 'alexnet' and train == 'general':
        return [(model.base_model.features, 1.0),  (model.feature_layers, 1.0),
            (model.fc, 1.0 * fc_weight), (model.discriminator, 1.0 * disc_weight)]
    elif name == 'caffenet' and train == 'general':
        return [(model.base_model.features, 1.0),  (model.base_model.classifier, 1.0),
            (model.base_model.class_classifier, 1.0 * fc_weight), (model.discriminator, 1.0 * disc_weight)]
    elif name == 'resnet' and train == 'general':
        return [(model.base_model.conv1, 1.0), (model.base_model.bn1, 1.0), (model.base_model.layer1, 1.0), 
                (model.base_model.layer2, 1.0), (model.base_model.layer3, 1.0), (model.base_model.layer4, 1.0), 
                (model.base_model.fc, 1.0 * fc_weight), (model.discriminator, 1.0 * disc_weight)]

def get_optimizer(model, init_lr, momentum, weight_decay, feature_fixed=False, nesterov=False, per_layer=False):
    if feature_fixed:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        if per_layer:
            if not isinstance(model, list):
                raise ValueError('Model must be a list type.')
            optimizer = optim.SGD(
                [{'params': model_.parameters(), 'lr': init_lr*alpha} for model_, alpha in model],
                lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
                                   
        else:
            params_to_update = model.parameters()
            optimizer = optim.SGD(
                params_to_update, lr=init_lr, momentum=momentum, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    return optimizer

schedulers_map = {
    'step': StepLR,
    'exponential': ExponentialLR,
    'inv': inv_lr_scheduler
}

def get_scheduler(name):
    if name not in schedulers_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_scheduler_fn(**kwargs):
        return schedulers_map[name](**kwargs)

    return get_scheduler_fn
    

def train_to_get_label(train, clustering):
    if train == 'deepall':
        return [False, False]
    elif train == 'general' and clustering == True:
        return [False, True]
    elif train == 'general' and clustering != True:
        return [True, False]
    else: 
        raise ValueError('Name of train unknown %s' % train)

        
from train import deepall, general
train_map = {
    'deepall': deepall.train,
    'general': general.train
}    

def get_train(name):
    if name not in train_map:
        raise ValueError('Name of train unknown %s' % name)
    def get_train_fn(**kwargs):
        return train_map[name](**kwargs)
    return get_train_fn

def get_disc_dim(name, clustering, domain_num, clustering_num):
    if name == 'deepall':
        return None
    elif name == 'general' and clustering == True:
        return clustering_num
    elif name == 'general' and clustering != True:
        return domain_num
    else:
        raise ValueError('Name of train unknown %s' % name)
    
def copy_weights(net_from, net_to):
    for m_from, m_to in zip(net_from.modules(), net_to.modules()):
        if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()
    return net_from, net_to