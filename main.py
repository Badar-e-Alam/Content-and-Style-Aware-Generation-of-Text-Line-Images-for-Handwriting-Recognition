import torch
from torch.utils.data import DataLoader
import tqdm
from parameters import *
from data_set import CustomImageDataset
import matplotlib.pyplot as plt
from loss import CER

from decoder import Decorder
from encoder_vgg import Encoder

# from models import Visual_encoder, TextEncoder_FC
from helper import pad_str, decoding
from torch import optim
import numpy as np
import time

# from encoder_vgg import Encoder

if __name__ == "__main__":
    """
    dataset:- returns the Images and Label in form of the batch_size and num_example
    pad_str:- Function is used the pad the strings in the label so each of the string has the equal length
              Basically pad the string with the empety size character until the maximum size is being achieved
    Str



    """

    TextDatasetObj = CustomImageDataset()
    no_workers = batch_size // num_example
    dataset = torch.utils.data.DataLoader(
        TextDatasetObj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=no_workers,
    )
    decoder_net = Decorder().to(device)
    encoder_net = Encoder().to(device)

    trainable_parameter = sum(
        param.numel() for param in encoder_net.parameters() if param.requires_grad
    )
    trainable_parameter_decoder = sum(
        param.numel() for param in decoder_net.parameters() if param.requires_grad
    )
    print(
        f" Encoder parameters = {trainable_parameter/ 1e6:.2f}.Millions \n Decoder paramters ={trainable_parameter/ 1e6:.2f}. Millions"
    )
    for Image, Label in tqdm.tqdm(dataset):
        label = pad_str(Label)
        print(f"{len(label)=} {len(label[0][0])=}")
        Str2Index = decoding(label=label, decoder=encoder)
        concate = torch.stack(Str2Index, dim=0)

        print(f"{concate[0].shape}")
        V_out = encoder_net(Image[0].to(device).unsqueeze(1))
        T_out = decoder_net.forward(concate[0].to(device), V_out)
        break

    # train_size = int(0.8 * (len(dataset)))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # main(
    #     train_loader=train_data_loader,
    #     test_loader=test_data_loader,
    #     num_writers=NUM_WRITERS,
    # )

    # max_str = 0
    # str_2_int = []
    # label = list()
    # global_step = len(dataset)
    # V_encoder = Visual_encoder()
    # V_encoder.to(device=device)
    # T_encoder = TextEncoder_FC()
    # T_encoder.to(device=device)
    # # for i in tqdm.tqdm(range(10)):
    # for img, label in tqdm.tqdm(dataset):
    #     label = pad_str(label)
    #     stio_ten = decoding(
    #         label=label, decoder=encoder
    #     )  # (number_example, batch_size,text_len_max)
    #     # stio_ten = encoding(label=label, encoder=encoder)
    #     concate = torch.stack(stio_ten, dim=0)
    #     for sample in range(num_example):
    #         # (batch,512,Img,img_height,img_width)
    #         # image = torch.transpose(
    #         #     torch.transpose(img[sample], 1, 3), 2, 3
    #         # )  # [8, 342, 2270, 3] [8,3,342,2270]
    #         image = img[sample].to(device=device)
    #         print("image shape", image.shape)
    #         V_out = V_encoder(image.unsqueeze(1))
    #         T_out, ful_cat = T_encoder(concate[sample].to(device=device))
    #     break

    #     # for index in range(num_example):
    #     #     # print(img[index].shape)
    #     #     # print(stio_ten[index].size())
    #     #     V_out = V_encoder(img[index])
