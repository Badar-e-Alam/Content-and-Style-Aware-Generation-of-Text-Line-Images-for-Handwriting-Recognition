import torch
import Levenshtein as Lev
from parameters import *
import torch.nn.functional as F
from load_data import tokens


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.detach() == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        if true_dist.requires_grad:
            print("Error! true_dist should not requires_grad!")
        return self.criterion(x, true_dist)


def kl_divergence_loss(predicted_probs, target_probs):
    eps = 1e-8  # Small constant to avoid division by zero

    # Broadcast target_probs to match the shape of predicted_probs
    # target_probs = target_probs.unsqueeze(0).expand_as(predicted_probs)
    long_soft = torch.nn.Softmax(dim=1)
    # target_probs = target_probs.squeeze(2)
    predicted_probs = long_soft(predicted_probs)
    target_probs = long_soft(target_probs.float())
    # predicted_probs=predicted_probs.squeeze(2)

    # Apply logarithm to predicted probabilities
    log_predicted_probs = torch.log(predicted_probs + eps)
    # Calculate the element-wise KL divergence

    kl_div = F.kl_div(log_predicted_probs, target_probs.view(1, -1), reduction="sum")
    return kl_div


log_softmax = torch.nn.LogSoftmax(dim=-1)

crit = LabelSmoothing(len(vocab), tokens["PAD_TOKEN"], 0.4)


def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list


class CER:
    def __init__(self):
        self.ed = 1
        self.len = 1

    def add(self, pred, gt):
        pred_label = torch.topk(pred, 1, dim=-1)[1].squeeze(-1)  # b,t,83->b,t,1->b,t
        pred_label = pred_label.cpu().numpy()
        batch_size = pred_label.shape[0]
        eds = list()
        lens = list()
        for i in range(batch_size):
            pred_text = pred_label[i].tolist()
            gt_text = gt[i].cpu().numpy().tolist()

            gt_text = fine(gt_text)
            pred_text = fine(pred_text)
            for j in range(len(tokens)):
                gt_text = list(filter(lambda x: x != j, gt_text))
                pred_text = list(filter(lambda x: x != j, pred_text))
            gt_text = "".join([decoder[c] for c in gt_text])
            pred_text = "".join([decoder[c] for c in pred_text])
            ed_value = Lev.distance(pred_text, gt_text)
            eds.append(ed_value)
            lens.append(len(gt_text))
        self.ed += sum(eds)
        self.len += sum(lens)

    def fin(self):
        return 100 * (self.ed / self.len)
