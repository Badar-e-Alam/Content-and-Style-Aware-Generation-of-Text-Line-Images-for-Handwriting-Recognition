import torch
import torch.nn as nn
from helper import batch_size
from load_data import tokens

from parameters import device, number_feature, NUM_CHANNEL,vocab


class Visual_encoder(nn.Module):
    def __init__(self):
        super(Visual_encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=batch_size, out_channels=100, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.Conv2d(
                in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(
            in_channels=100, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        # self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        print("Shape of the Input in VGG network:-", x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.upsample1(x)
        # x=self.upsample2(x)
        # x=self.upsample3(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

    def forward(self, x):

        return self.model(x)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = NUM_CHANNEL
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    #'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # b,3,64,512 -> b,512,2,16
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # b,3,64,512 -> b,512,4,32
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # b,3,64,512 -> b,512,8,64
    'E': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # need to be updated
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # if pretrained:
    #     kwargs['init_weights'] = False
    #     model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    #     model_dict = model.state_dict()
    #     total_dict = model_zoo.load_url(model_urls['vgg19_bn'])
    #     #total_dict = torch.load(model_urls['vgg19_bn'])
    #     partial_dict = {k: v for k, v in total_dict.items() if k in model_dict}
    #     model_dict.update(partial_dict)
    #     model.load_state_dict(partial_dict)
    # else:
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        self.embed = nn.Embedding(len(vocab), embed_size)
        self.fc = nn.Sequential(
                nn.Linear(text_max_len*embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, number_feature)
                )
        '''embed content force'''
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):

        xx = self.embed(x) # b,t,embed
        batch_size = xx.shape[0]
        xxx = xx.view(batch_size, -1) # b,t*embed
        out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512

        ts = f_xs_shape[-1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        if width_reps==0:
                 width_reps=1 
        tensor_list = list()

        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long))
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)

            tensor_list.append(padding)
        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)
        return out, final_res