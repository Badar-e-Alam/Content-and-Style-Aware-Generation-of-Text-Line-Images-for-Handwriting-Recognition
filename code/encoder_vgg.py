import torch
from torch import nn

# from torch.autograd import Variable
from parameters import device, batch_size, MAX_CHARS, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL
import torch.nn.functional as F

# from models.vgg_tro_channel1 import vgg16_bn
from simple_resnet import ResidualBlock
from decoder import LayerNormLinearDropoutBlock
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # B, C, H, W = (
        #     batch_size,
        #     32,
        #     IMAGE_HEIGHT * scale_factor,
        #     IMAGE_WIDTH * scale_factor,
        # )
        self.num_layer = 6
        out_feature=1000
        self.num_heads = 2
        self.in_feature = IMG_HEIGHT * IMG_WIDTH * batch_size
        self.out_feature = MAX_CHARS
       
        weights_path="renet_weights/resnet34-333f7ec4.pth"
    # Load the model.
        self.resnet = models.resnet34(pretrained=False)

        # Load the weights store.
        weights_store = torch.load(weights_path)

        # Load the weights into the model.
        self.resnet.load_state_dict(weights_store)

        # self.resnet = Generator_Resnet(num_res_blocks=2).to(device)
        self.resnet = nn.Sequential(
            *[
                ResidualBlock(NUM_CHANNEL * 2 ** i, NUM_CHANNEL * 2 ** (i + 1))
                for i in range(self.num_layer)
            ]
        )
        #reshape resnet into the b,w and h*c 
        # self.visual_encoder = Visual_encoder().to(device)  # vgg
        #self.visual_encoder = ImageEncoder().to(device=device)
        self.norm_down = nn.Linear(
            out_feature, out_features=self.out_feature
        )
        self.upsample_norm = nn.Conv2d(batch_size, NUM_CHANNEL * self.num_heads, 1, 1,)

        self.block_with_attention = LayerNormLinearDropoutBlock(
            #batch,512,32,80 -> batch,32,H*feature
            in_features=IMG_HEIGHT*64,#)(batch_size*NUM_CHANNEL,IMG_WIDTH,IMG_HEIGHT*(2**(self.num_layer+1))), #batch,width,h*feature_dim,
            out_features=IMG_HEIGHT*64,
            num_heads=self.num_heads,
            dropout_prob=0.2,
            attention=True,
        )
        self.block_without_attention = LayerNormLinearDropoutBlock(
            in_features=IMG_HEIGHT*64,#IMG_HEIGHT * IMG_WIDTH * batch_size,
            out_features=IMG_HEIGHT*64,
            num_heads=self.num_heads,
            dropout_prob=0.2,
            attention=False,
        )
        self.norm = nn.LayerNorm(IMG_HEIGHT*64)

    def forward(self, x):
        resnet = self.resnet(  # shape matches
            x.permute(1,0,2,3)
        )  # resent   batch_size,outchannel,Hight , Width
        b,c,h,w=resnet.shape
        resnet=resnet.reshape(b,w,h*c)

        # resent=resent.view(batch_size,-1)
        # visual_encode = self.visual_encoder(x)  # visual encoder for positionin
        # # visual_encder=visual_encder.view(batch_size,-1)
        # resnet_shape = resnet.shape
        # resized_vis = F.interpolate(
        #     visual_encode, size=resnet_shape[2:], mode="bilinear", align_corners=False
        # )

        # repeat = resnet.size(0) // visual_encode.size(0)
        # if repeat == 0:
        #     repeat = resnet.size(0)
        #     resnet = resnet.repeat(*(repeat, 1, 1, 1))
        # else:

        #     resized_vis = resized_vis.repeat(*(repeat, 1, 1, 1))

        # combained_out = resnet + resized_vis  # combained before input
      
        # combained_out = F.interpolate(
        #     combained_out,
        #     size=[IMG_HEIGHT, IMG_WIDTH],
        #     mode="bilinear",
        #     align_corners=False,
        # )
        #batch,h,w,dim  one dim
        attention_block, norm_layer = self.block_with_attention(
            resnet
        )

        #down_sampled_norm = self.norm_down(norm_layer)
        # up_sample_norm=self.upsample_norm(down_sampled_norm.unsqueeze(1))
        # import pdb;pdb.set_trace()

        # down_sampled_norm = down_sampled_norm.repeat(
        #     attention_block.size(0) // down_sampled_norm.size(0), 1
        # )

        combained_attention = norm_layer+attention_block

        without_attention, _ = self.block_without_attention(norm_layer+attention_block)
        combained_with_attention = combained_attention + without_attention
        final_norm = self.norm(combained_with_attention)  # 4,32
        import pdb;pdb.set_trace()
        return final_norm
