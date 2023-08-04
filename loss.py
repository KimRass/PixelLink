import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import get_textbox_masks, get_areas

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)

# Instance-Balanced Cross-Entropy Loss
# "For a given image with $N$ text instances, all instances are treated equally by giving a same weight
# to everyone of them, denoted as $B_{i}$. For the $i$-th instance with $area = S_{i}$,
# every positive pixels within it have a weight of $w_{i} = B_{i} / S_{i}$."
# "$B_{i} = S / N$"
# "$S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$"

NEG_POS_RATIO = 3 # "$r"


# "Online Hard Example Mining (OHEM) is applied to select negative pixels. $r * S$ negative pixels
# with the highest losses are selected, by setting their weights to ones. $r$ is the negative-positive ratio
# and is set to 3 as a common practice."

# "The loss on pixel classification task is: $L_{pixel} = \frac{1}{1 + r)S}WL_{pixel_CE}$
# where $L_{pixel_CE} is the matrix of Cross-Entropy loss on text/non-text prediction. As a result,
# pixels in small instances have a higher weight, and pixels in large instances have a smaller weight.
# However, every instance contributes equally to the loss."

# "Losses for positive and negative links"
n_neg_pixels = NEG_POS_RATIO * tot_area

class InstanceBalancedCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        crit = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pixel_pred, pixel_gt, link_pred, link_gt):
        pixel_pred = torch.randn(4, 1, 2000, 1500)
        pixel_pred = torch.randn(4, 2, 2000, 1500)
        ce_loss = crit(pixel_pred.permute(0, 2, 3, 1).reshape(-1, 2), pixel_gt.view(-1))
        ce_loss.shape

