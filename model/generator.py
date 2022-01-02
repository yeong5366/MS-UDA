import torch
import torch.nn as nn
import torchvision.models as models

class Generator(nn.Module):

    def __init__(self, sensor = 'rgb', num_layers=50, recon = False):
        super(Generator, self).__init__()

        if num_layers ==50:
            resnet_raw_model = models.resnet50(pretrained=True)
        elif num_layers == 101:
            resnet_raw_model = models.resnet101(pretrained=True)
        elif num_layers == 152:
            resnet_raw_model = models.resnet152(pretrained=True)

        if sensor == 'rgb':
            self.conv1 = resnet_raw_model.conv1
        elif sensor =='thermal':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model.conv1.weight.data, dim=1), dim=1)

        self.bn1 = resnet_raw_model.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = resnet_raw_model.layer1
        self.layer2 = resnet_raw_model.layer2
        self.layer3 = resnet_raw_model.layer3
        self.layer4 = resnet_raw_model.layer4

        if recon : self.recon = recon

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
