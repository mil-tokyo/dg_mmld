from torchvision.models import resnet18
from model.Discriminator import Discriminator
import torch.nn as nn
import torch.nn.init as init

def resnet(num_classes, num_domains=None, pretrained=True):
    model = resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model

class DGresnet(nn.Module):
    def __init__(self, num_classes, num_domains, pretrained=True, grl=True):
        super(DGresnet, self).__init__()
        self.num_domains = num_domains
        self.base_model = resnet(num_classes=num_classes, pretrained=pretrained)
        self.discriminator = Discriminator([512, 1024, 1024, num_domains], grl=grl, reverse=True)
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        output_class = self.base_model.fc(x)
        output_domain = self.discriminator(x)
        return output_class, output_domain
        
    def features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def conv_features(self, x) :
        results = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # results.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        results.append(x)
        x = self.base_model.layer2(x)
        results.append(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # results.append(x)
        return results        
    
    def domain_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        return x.view(x.size(0), -1)