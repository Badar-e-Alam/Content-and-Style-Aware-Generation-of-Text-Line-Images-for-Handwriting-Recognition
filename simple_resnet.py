from torch import nn
import torch
from parameters import *


class ResNET(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(ResNET, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResNET(
                    in_dim,
                    out_dim,
                )
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    net = ResBlocks(num_blocks=6, in_dim=1, out_dim=512)
    net.to(device)
    input = torch.rand(size=(20, 1, 25, 10), device=device)
    out = net(input)
    print(out.shape)
