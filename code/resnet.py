import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import device
from models import ImageEncoder, TextEncoder_FC

# from text_style import TextEncoder_FC
from decoder import Decorder
from encoder_vgg import Encoder
from block import Generator_Resnet
from parameters import batch_size
from load_data import IMG_HEIGHT, IMG_WIDTH
from simple_resnet import ResidualBlock, AdaLN


class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        # define the number of layers
        self.n_layers = 5
        self.final_size = 1024
        in_dim = batch_size
        out_dim = 16
        resnet_out = 512
        self.ff_cc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=7,
            stride=1,
            padding="same",
        )
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(out_dim * 2 ** i, out_dim * 2 ** (i + 1))
                for i in range(self.n_layers)
            ]
        )
        self.cnn_f = nn.Conv2d(
            resnet_out, self.final_size, kernel_size=7, stride=1, padding="same"
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        feat = self.res_blocks(self.ff_cc(x))
        out = self.cnn_f(feat)
        return out.squeeze(-1).squeeze(
            -1
        )  # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def calc_dis_fake_loss(self, input_fake):
        resp_fake = self.forward(input_fake)
        label = torch.zeros(resp_fake.shape).to(device)

        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        resp_real = self.forward(input_real)
        label = torch.ones(resp_real.shape).to(device)

        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        resp_fake = self.forward(input_fake.permute(1, 0, 2, 3))
        label = torch.ones(resp_fake.shape).to(device)

        fake_loss = self.bce(resp_fake, label)
        return fake_loss


class WriterClaModel(nn.Module):
    """
        WriterclaModel  classification method for the writer class
            
        contain covultional layer at the start and end I guess to make intput and output data 
        compatible shape and size with other modules of the code. Then the inner part contains
        residual block of code  dependes on number of layer.           
    """

    def __init__(self, num_writers) -> None:
        super(WriterClaModel, self).__init__()
        self.n_layers = 5
        in_dim = 512
        out_dim = 16
        self.num_writers = num_writers
        #
        self.cnn_f = nn.Conv2d(
            in_channels=in_dim,
            out_channels=16,
            kernel_size=3,
            padding=1,
            stride=1
            # padding_mode="reflect",
        )
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(out_dim * 2 ** i, out_dim * 2 ** (i + 1))
                for i in range(self.n_layers)
            ]
        )
        self.ff_cc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=self.num_writers,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        cnn_f = self.cnn_f(x)
        resnet = self.res_blocks(cnn_f)
        ff_cc = self.ff_cc(resnet)

        loss = self.cross_entropy(ff_cc.view(ff_cc.size(0), -1), y.view(-1).long())
        return loss


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder().to(device)
        self.enc_text = TextEncoder_FC(text_max_len).to(device)
        self.generator = Generator_Resnet().to(device)
        self.linear_mix = nn.Conv2d(4, 2, 1, 1).to(device)
        self.linear_mix_final = nn.Linear(1024, 512)

        self.down_sampling = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1
        )

    def generate(self, content, adain_params):
        # decode content and style codes to an image
        assign_adain_params(adain_params, self.generator)

        return self.generator(content)

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        down_sampling = self.down_sampling(feat_embed)
        feat_mix = torch.cat([feat_xs, down_sampling], dim=0)  # b,1024,8,27

        ff = self.linear_mix(feat_mix.permute(1, 0, 2, 3))  # b,8,27,1024->b,8,27,512
        return ff

    def mix_final(self, feat_xs, feat_embed):

        feat_mix = torch.cat([feat_xs, feat_embed], dim=1)  # b,1024,8,27
        f = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix_final(f)  # b,8,27,1024->b,8,27,512

        return ff


class RecModel(nn.Module):
    def __init__(self):
        super(RecModel, self).__init__()
        self.enc = Encoder().to(device)
        self.dec = Decorder().to(device)

    def forward(self, image, text):
        visual_out = self.enc(image.to(device))
        text_visual = self.dec(visual_out, text, image.shape)
        return text_visual
