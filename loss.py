import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Instance-Balanced Cross-Entropy Loss
# "For a given image with $N$ text instances, all instances are treated equally by giving a same weight
# to everyone of them, denoted as $B_{i}$. For the $i$-th instance with $area = S_{i}$,
# every positive pixels within it have a weight of $w_{i} = B_{i} / S_{i}$."
# "$B_{i} = S / N$"
# "$S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$"

# "Online Hard Example Mining (OHEM) (Shrivastava, Gupta, and Girshick 2016) is applied to select negative pixels. More specifically, r   S negative pixels with the highest losses are selected, by setting their weights to ones. r is the negative-positive ratio and is set to 3 as a common practice. The above two mechanisms result in a weight matrix, denoted by W, for all positive pixels and selected negative"
NEG_POS_RATIO = 3


def get_weight_map(bboxes):
    n_insts = len(bboxes) # $N$
    bboxes["area"] = (bboxes["xmax"] - bboxes["xmin"]) * (bboxes["ymax"] - bboxes["ymin"]) # $S_{i}$
    tot_area = bboxes["area"].sum() # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$
    avg_area = tot_area / n_insts # $B_{i} = S / N$
    bboxes["weight"] = avg_area / bboxes["area"] # $w_{i} = B_{i} / S_{i}$
    bboxes.head()

n_neg_pixels = NEG_POS_RATIO * tot_area

class InstanceBalancedCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        crit = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pixel_pred, pixel_gt):
        pixel_pred = torch.randn(4, 2, 2000, 1500)
        ce_loss = crit(pixel_pred.permute(0, 2, 3, 1).reshape(-1, 2), pixel_gt.view(-1))
        ce_loss.shape