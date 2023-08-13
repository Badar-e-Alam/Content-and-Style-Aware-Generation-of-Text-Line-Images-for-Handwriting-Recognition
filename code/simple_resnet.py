import torch
import torch.nn.functional as F
import os
from helper import *
import cv2
from torch import nn


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class AdaLN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.rho, 0.9)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.beta, 0.0)

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.adaln = AdaLN(out_channels)  # Assuming you have defined AdaLN separately

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.adaln(x)

        x += residual
        x = self.leaky_relu(x)
        return x


def write_image(
    xg,
    pred_label,
    gt_img,
    gt_label,
    tr_imgs,
    xg_swap,
    pred_label_swap,
    gt_label_swap,
    title,
    num_tr=2,
):
    folder = "imgs"
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(
        -1
    )  # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(
        -1
    )  # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr * IMG_HEIGHT, -1)
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        tar_swap = xg_swap[i].squeeze()
        src = normalize(src)
        gt = normalize(gt)
        tar = normalize(tar)
        tar_swap = normalize(tar_swap)
        gt_text = gt_label[i].tolist()
        gt_text_swap = gt_label_swap[i].tolist()
        pred_text = pred_label[i].tolist()
        pred_text_swap = pred_label_swap[i].tolist()

        gt_text = fine(gt_text)
        gt_text_swap = fine(gt_text_swap)
        pred_text = fine(pred_text)
        pred_text_swap = fine(pred_text_swap)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x != j, gt_text))
            gt_text_swap = list(filter(lambda x: x != j, gt_text_swap))
            pred_text = list(filter(lambda x: x != j, pred_text))
            pred_text_swap = list(filter(lambda x: x != j, pred_text_swap))

        gt_text = "".join([index2letter[c - num_tokens] for c in gt_text])
        gt_text_swap = "".join([index2letter[c - num_tokens] for c in gt_text_swap])
        pred_text = "".join([index2letter[c - num_tokens] for c in pred_text])
        pred_text_swap = "".join([index2letter[c - num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(
            gt_text_img,
            gt_text,
            (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            gt_text_img_swap,
            gt_text_swap,
            (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            pred_text_img,
            pred_text,
            (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            pred_text_img_swap,
            pred_text_swap,
            (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        out = np.vstack(
            [
                src,
                gt,
                gt_text_img,
                tar,
                pred_text_img,
                gt_text_img_swap,
                tar_swap,
                pred_text_img_swap,
            ]
        )
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder + "/" + title + ".png", final_out)
