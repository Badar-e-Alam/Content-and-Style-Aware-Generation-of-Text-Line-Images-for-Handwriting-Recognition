import torch
from torch import nn
from attention import MultiHeadAttention, MultiHead_CrossAttention,Attention,CrossAttention
from models import TextEncoder_FC
import  torch.nn.functional as F
from parameters import (
    MAX_CHARS,
    batch_size,
    device,
    number_feature,
    NUM_CHANNEL,
    IMG_HEIGHT,
    IMG_WIDTH,
)

input_feature = NUM_CHANNEL * IMG_WIDTH * IMG_HEIGHT


class Decorder(torch.nn.Module):
    def __init__(
        self, in_feature=IMG_HEIGHT * IMG_WIDTH, out_feature=MAX_CHARS, dropout=0.3
    ):
        super().__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.TextStyle = TextEncoder_FC(MAX_CHARS).to(device)

        self.linear_upsampling = nn.Linear(number_feature, input_feature)
        self.linear_downsampling = nn.Conv2d(
            in_channels=IMG_HEIGHT*512, out_channels=IMG_HEIGHT*64,kernel_size=1,padding=1,stride=1
        )
        ##self.upsample_norm=nn.Conv2d(batch_size*num_heads,NUM_CHANNEL*num_heads,1,1,)

        self.block_with_attention = LayerNormLinearDropoutBlock(
            in_features=IMG_HEIGHT*64,
            out_features=IMG_WIDTH,
            num_heads=2,
            dropout_prob=0.2,
            attention=True,
        )
        self.block_without_attention = LayerNormLinearDropoutBlock(
            in_features=IMG_HEIGHT*64,
            out_features=IMG_WIDTH,
            num_heads=2,
            dropout_prob=0.2,
            attention=False,
        )
        self.cross_attention = CrossAttention(
        query_dim=IMG_HEIGHT*64,key_dim=IMG_HEIGHT*64,value_dim=IMG_HEIGHT*64
           
        )
        self.drop = nn.Dropout(self.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, encoder_out, text_style_content=None, img_shape=None,
    ):

        # char_embedding= 2,5440 global_net=[2,64,10,100]  batch,output,H,W
        _, global_net = self.TextStyle(text_style_content, img_shape)
        b,c,h,w=global_net.shape
        global_net=global_net.reshape(b,h*c,w)
        import pdb;pdb.set_trace()
        global_net=self.linear_downsampling(global_net.permute(1,0,2))
        #char_upsampling = self.linear_upsampling(char_embedding)
        # if char_upsampling.reshape(-1).shape!=global_net.reshape(-1).shape:
        #    last_dim=char_upsampling.size(1)//(global_net.size(0)* global_net.size(1)* global_net.size(2))
        #    char_upsampling= char_upsampling.view(global_net.size(0), global_net.size(1), global_net.size(2),last_dim)
        #    char_upsampling = F.interpolate( char_upsampling,  size=[IMG_HEIGHT, IMG_WIDTH],  mode="bilinear", align_corners=False,  )
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # txt_style = global_net + char_upsampling.view(
        #     global_net.size(0),
        #     global_net.size(1),
        #     global_net.size(2),
        #     global_net.size(3),
        # )
        ## layer norm has the same shape are hte txt_style and global_net [2,64000]
        # attention_block=2,85

        attention_block, layer_norm = self.block_with_attention(
            global_net.permute(1,2,0)
        )
        # norm_down_sample = self.linear_downsampling(layer_norm)
        # norm_down_sample = norm_down_sample.repeat(
        #     attetion_block.size(0) // batch_size, 1
        # )  # mask for  text
        import pdb;pdb.set_trace()
        attention_norm = attention_block + layer_norm
        block_without_attention, _ = self.block_without_attention(attention_norm)

        combained_without_attention = block_without_attention + attention_norm


        # upsample_norm=self.upsample_norm(norm.unsqueeze(1)).squeeze(1)
        #norm = norm.repeat(encoder_out.size(0) // norm.size(0), 1)
        import pdb;pdb.set_trace()

        cross_attention = self.cross_attention(combained_without_attention, encoder_out)
        drop_out = self.drop(cross_attention)
        # norm = norm.repeat(drop_out.size(0) // (batch_size + batch_size), 1)
        combained_without_attention = drop_out + norm

        block_without_attention2, _ = self.block_without_attention(
            combained_without_attention
        )
        final_combained = block_without_attention2 + combained_without_attention

        soft_max = self.softmax(final_combained)
        return final_combained


class LayerNormLinearDropoutBlock(nn.Module):
    def __init__(
        self, in_features, out_features, num_heads, dropout_prob=0.1, attention=False
    ):
        super(LayerNormLinearDropoutBlock, self).__init__()
        self.attention = attention
        # Define the layer norm, linear layer, and dropout modules
        if self.attention:
            self.layer_norm = nn.LayerNorm(in_features) #N,C,H,W
            self.atten =Attention(in_features)
        else:
            self.layer_norm = nn.LayerNorm(in_features)
            self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Apply layer norm to the input tensor
        layer_norm = self.layer_norm(x)
        if self.attention:
            x = self.atten(layer_norm)
        else:
            # Apply linear transformation to the input tensor

            x = self.linear(layer_norm)

        # Apply dropout to the output of the linear layer
        x = self.dropout(x)

        return x, layer_norm
