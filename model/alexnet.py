import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.Discriminator import Discriminator
from torchvision.models import AlexNet

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def alexnet(num_classes, num_domains=None, pretrained=True):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        print('Load pre trained model')
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model

class DGalexnet(nn.Module):
    def __init__(self, num_classes, num_domains, pretrained=True, grl=True):
        super(DGalexnet, self).__init__()
        self.num_domains = num_domains
        self.base_model = alexnet(num_classes, pretrained=pretrained)
        self.discriminator = Discriminator([4096, 1024, 1024, num_domains], grl=grl, reverse=True)
        self.feature_layers = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        self.fc = list(self.base_model.classifier.children())[-1]
        
    def forward(self, x):
        x = self.base_model.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.feature_layers(x)
        output_class = self.fc(x)
        output_domain = self.discriminator(x)
        return output_class, output_domain

    def features(self, x):
        x = self.base_model.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.feature_layers(x)
        return x

    def conv_features(self, x) :
        results = []
        for i, model in enumerate(self.base_model.features):
            x = model(x)
            if i in {4, 7}:
                results.append(x)
        return results
    
    def domain_features(self, x):
        for i, model in enumerate(self.base_model.features):
            x = model(x)
            if i == 7:
                break
        return x.view(x.size(0), -1)