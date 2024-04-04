from torch import nn

# from torch.autograd import Variable
import numpy as np
from parameters import *
import torch.functional as F
from attention import MultiHeadAttention

# from models.vgg_tro_channel1 import vgg16_bn
from models import Visual_encoder
from block import *
from decoder import LayerNormLinearDropoutBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        B, C, H, W = (
            batch_size,
            32,
            IMAGE_HEIGHT * scale_factor,
            IMAGE_WIDTH * scale_factor,
        )
        self.num_heads = 4
        head_size = 200
        print(f"channel:-{C=},Hight:- {H=} width:- {W=} Batch:- {B=}")
        self.in_feature = 32 * IMAGE_HEIGHT * scale_factor * IMAGE_WIDTH * scale_factor
        self.out_feature = 128
        self.resnet = Generator_Resnet(class_num=2, num_res_blocks=2).to(device)
        self.visual_encoder = Visual_encoder().to(device)  # vgg
        self.linear_downsampling = nn.Linear(
            in_features=self.in_feature, out_features=self.out_feature
        )
        self.block_with_attention = LayerNormLinearDropoutBlock(
            in_features=self.in_feature,
            out_features=self.out_feature,
            num_heads=2,
            dropout_prob=0.2,
            attention=True,
        )
        self.block_without_attention = LayerNormLinearDropoutBlock(
            in_features=self.in_feature,
            out_features=self.out_feature,
            num_heads=2,
            dropout_prob=0.2,
            attention=False,
        )
        self.norm = nn.LayerNorm(self.out_feature)

    def forward(self, x):
        resent = self.resnet(x)  # resent   batch_size,outchannel,Hight , Width

        # resent=resent.view(batch_size,-1)
        visual_encder = self.visual_encoder(x)  # visual encoder for positionin
        # visual_encder=visual_encder.view(batch_size,-1)
        print(
            f"Shape of the resent output{resent.shape} and Vgg output shape{visual_encder.shape}"
        )
        combained_out = resent + visual_encder  # combained before input
        attention_block, norm_layer = self.block_with_attention(
            combained_out.view(combained_out.size(0), -1)
        )
        down_sampled_norm = self.linear_downsampling(norm_layer)
        down_sampled_norm = down_sampled_norm.repeat(
            attention_block.size(0) // batch_size, 1
        )
        combained_attention = down_sampled_norm + attention_block
        without_attention, _ = self.block_without_attention(combained_attention)
        combained_with_attention = combained_attention + without_attention
        final_norm = self.norm(combained_with_attention)
        print("End of encoder")
        return final_norm
