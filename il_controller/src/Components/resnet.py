import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from Data_processing import global_params
import pdb
import numpy as np

config = global_params.config

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if config.disable_bn_in_resnet == False:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if config.disable_bn_in_resnet == False:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if not config.disable_bn_in_resnet:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if not config.disable_bn_in_resnet:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if not config.disable_bn_in_resnet:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, fnsize=1, ap_kernelsize=3, ap_pad=0, ip_channels=3,
                 num_classes=config.num_resnet_output):  # ap_kernelsize changed from 7 to 3
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.num_layers = config.Num_resnet_layers

        self.conv1 = None
        if self.num_layers == 3 or self.num_layers == 2:
            self.conv1 = nn.Conv2d(ip_channels, 64, kernel_size=7, stride=2, padding=3,  # changing stride from 2 to 1
                                   bias=False)  # changing no of input channels (originally 3 in resnet)
        elif self.num_layers == 4:
            self.conv1 = nn.Conv2d(ip_channels, 64, kernel_size=7, stride=1, padding=3,  # changing stride from 2 to 1
                                   bias=False)  # changing no of input channels (originally 3 in resnet)

        self.bn1 = nn.BatchNorm2d(64, track_running_stats=config.track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if self.num_layers == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # stide changed from 2 to 1
        elif self.num_layers > 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # stide changed from 2 to 1

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        final_channel = 128

        self.layer3 = None
        if self.num_layers >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            final_channel = 256

        # pdb.set_trace()
        self.layer4 = None
        if self.num_layers == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            final_channel = 512

        self.avgpool = nn.AvgPool2d(ap_kernelsize, stride=1, padding=ap_pad)

        # self.fc = nn.Linear(final_channel * block.expansion *
        #                   fnsize * fnsize, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                m.bias.data.normal_(0, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not config.disable_bn_in_resnet:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, track_running_stats=config.track_running_stats),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if config.disable_bn_in_resnet == False:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.layer3:
            x = self.layer3(x)
        if self.layer4:
            x = self.layer4(x)

        final_image = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = self.fc(x)
        return x, final_image


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model
