from model.resnet import resnet50
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2)
            )

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class Mynet(nn.Module):
    """
    基于resnet50的验证码识别网络模型
    """

    def __init__(self, classes, numbers, pretrained):
        super(Mynet, self).__init__()
        self.classes = classes
        self.numbers = numbers
        self.head = resnet50(pretrained=pretrained)
        self.classfy = self._make_layer(classes, numbers)

    def forward(self, x):
        y = self.head(x)  # y.shape = [1, 1024, 13, 19]
        y = self.classfy(y) # u.shape = [1, 248]
        y = y.view(-1, self.numbers * self.classes)
        return y

    def _make_layer(self, classes, numbers):
        layers = nn.Sequential(
            Bottleneck(1024, 512, downsample=self._make_downsample(1024, 512)),
            Bottleneck(512, 256, downsample=self._make_downsample(512, 256)),
            Bottleneck(256, 64, downsample=self._make_downsample(256, 64)),
            SPPLayer(4),
            nn.Linear(1920, numbers * classes, bias=True),
            nn.Dropout(0.5, inplace=True),
        )

        return layers

    def _make_downsample(self, inplane, outplane):
        downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplane),
        )
        return downsample


class Net(nn.Module):
    """
    a simple model for captcha detect
    """

    def __init__(self, classes, numbers):
        super(Net, self).__init__()
        self.classes = classes
        self.numbers = numbers

        # 第一层神经网络
        # nn.Sequential: 将里面的模块依次加入到神经网络中
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3通道变成16通道，图片：44*140
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：22*70
        )
        # 第2层神经网络
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), # 16通道变成64通道，图片：20*68
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：10*34
        )
        # 第3层神经网络
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3), # 16通道变成64通道，图片：8*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片：4*16
        )
        # 第4层神经网络
        self.fc1 = nn.Sequential(
            nn.Linear(4*16*128, 1024),
            nn.Dropout(0.5),  # drop 20% of the neuron
            nn.ReLU()
        )
        # 第5层神经网络
        self.fc2 = nn.Linear(1024, self.classes*self.numbers) # 6:验证码的长度， 37: 字母列表的长度

    #前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x