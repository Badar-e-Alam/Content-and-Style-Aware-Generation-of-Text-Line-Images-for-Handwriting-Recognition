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
            self.model += [ResNET(in_dim, out_dim,)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

def write_image(xg, pred_label, gt_img, gt_label, tr_imgs, xg_swap, pred_label_swap, gt_label_swap, title, num_tr=2):
    folder = 'imgs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr*IMG_HEIGHT, -1)
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
            gt_text = list(filter(lambda x: x!=j, gt_text))
            gt_text_swap = list(filter(lambda x: x!=j, gt_text_swap))
            pred_text = list(filter(lambda x: x!=j, pred_text))
            pred_text_swap = list(filter(lambda x: x!=j, pred_text_swap))


        gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
        gt_text_swap = ''.join([index2letter[c-num_tokens] for c in gt_text_swap])
        pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
        pred_text_swap = ''.join([index2letter[c-num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(gt_text_img_swap, gt_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img_swap, pred_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img, gt_text_img_swap, tar_swap, pred_text_img_swap])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder+'/'+title+'.png', final_out)
