import torch
import torch.nn as nn
from parameters import *
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import cv2

from seq2sqe import *
from decoder import *


class Visual_encoder(nn.Module):
    def __init__(self) -> None:
        super(Visual_encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=100, kernel_size=3, stride=1, padding=1
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
            in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        print("Shape of the Input in VGG network:-", x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.upsample1(x)
        # x=self.upsample2(x)
        # x=self.upsample3(x)
        return x


class TextEncoder_FC(nn.Module):
    def __init__(self) -> None:
        super(TextEncoder_FC, self).__init__()
        """
         self.embed = Apply the embedding layer on the text tensor(2,85) -> (batch_size,max_text_len) -> out= (batch_size,max_len,embedding_size)
         xx = (batch_size, max_len_embedding_size)
         xxx = reshape the embedding output  from (batch_size,max_len_text,embedding_size) -> (batch_size,max_len*embedding_size) 
         out = Contained the output of the text style_network out_dim -> (batch_size,4096)

         xx_new =  apply the Linear layer on the embedding output 

        """
        self.embed = nn.Embedding(len(vocab), embedding_size)  # 81,64
        self.fc = nn.Sequential(
            nn.Flatten(),  # flatten the input tensor to a 1D tensor
            nn.Linear(text_max_len * embedding_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 5440),
        )
        self.linear = nn.Linear(
            embedding_size * text_max_len, embedding_size * text_max_len
        )  # 64,512
        self.linear1 = nn.Linear(embedding_size, embedding_size * text_max_len)

    def forward(self, x):
        xx = self.embed(x)  # b,t,embed

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1)  # b,t*embed
        out = self.fc(xxx)

        """embed content force"""
        xx_new = self.linear(xx.view(2, -1)).view(
            xx.size(0), xx.size(1), xx.size(2)
        )  # b, text_max_len, 512

        ts = xx_new.shape[1]  # b,512,8,27
        height_reps = IMAGE_HEIGHT  # 8 [-2]
        width_reps = max(1, IMAGE_WIDTH // ts)  # [-2] 27
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i : i + 1]]  # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = IMAGE_WIDTH % ts
        if padding_reps:
            embedded_padding_char = self.embed(
                torch.full((1, 1), 2, dtype=torch.long, device=device)
            )
            # embedded_padding_char = self.linear1(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(
            tensor_list, dim=1
        )  # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(
            2
        )  # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)
        return out, final_res
