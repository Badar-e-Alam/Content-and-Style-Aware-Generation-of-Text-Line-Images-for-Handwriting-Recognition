import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import number_feature
from helper import batch_size,device

class AdaLN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.rho, 0.9)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.beta, 0.0)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation=F.relu,
        num_feature=number_feature,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        self.activation = activation
        self.adaln = AdaLN(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.adaln(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class Generator_Resnet(nn.Module):
    def __init__(self, num_res_blocks=2, norm_layer=AdaLN, activation=F.leaky_relu):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.norm_layer = norm_layer
        self.activation = activation
        self.conv1 = nn.Conv2d(batch_size, 64, kernel_size=3, stride=1, padding=1,)
        self.bn1 = norm_layer(64)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    64, 64, norm_layer=self.norm_layer, activation=self.activation
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1,)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Third convolutional module
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Fourth convolutional module
        self.conv5 = nn.Conv2d(128, batch_size, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        # Final activation layer
        self.tanh = nn.Tanh()

    def forward(self, x):

        print("shape of  the input image in generator resnet :-", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.res_blocks(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x=  self.conv3(x)

        x=self.relu3(x)

        x=  self.conv4(x)
        x=self.relu4(x)

        x=  self.conv5(x)
        x=self.relu5(x)
        x = self.tanh(x)

        return x

