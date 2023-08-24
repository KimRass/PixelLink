import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# from data import get_textbox_masks, get_areas

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)

IMG_SIZE = 512
# "$r$ is the negative-positive ratio and is set to 3 as a common practice."
NEG_POS_RATIO = 3


class InstanceBalancedCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def _apply_ohem(self, ce_loss, pixel_weight, tot_area):
        temp_loss = ce_loss.clone()
        temp_loss[pixel_weight != 0] = 0
        sorted_indices = torch.sort(temp_loss, descending=True)[1]
        n_neg_pixels = NEG_POS_RATIO * tot_area
        highest_loss_neg_pixels = sorted_indices[: n_neg_pixels]
        pixel_weight[highest_loss_neg_pixels] = 1
        # "Online Hard Example Mining (OHEM) is applied to select negative pixels. $r * S$ negative pixels
        # with the highest losses are selected, by setting their weights to ones."
        return pixel_weight

    # "For a given image with $N$ text instances, all instances are treated equally by giving a same weight
    # to everyone of them, denoted as $B_{i}$. For the $i$-th instance with $area = S_{i}$,
    # every positive pixels within it have a weight of $w_{i} = B_{i} / S_{i}$."
    # "$B_{i} = S / N$"
    # "$S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$"
    def forward(self, pixel_pred, pixel_gt, pixel_weight):
        pixel_pred = rearrange(pixel_pred, pattern="b c h w -> (b h w) c")
        pixel_gt = rearrange(pixel_gt, pattern="b c h w -> (b h w) c").squeeze()
        pixel_weight = rearrange(pixel_weight, pattern="b c h w -> (b h w) c").squeeze()

        # "The loss on pixel classification task is: $L_{pixel} = \frac{1}{1 + r)S}WL_{pixel_CE}$
        # where $L_{pixel_CE} is the matrix of Cross-Entropy loss on text/non-text prediction.
        # As a result, pixels in small instances have a higher weight, and pixels in large instances
        # have a smaller weight. However, every instance contributes equally to the loss."
        ce_loss = self.ce(pixel_pred, pixel_gt)
        tot_area = pixel_gt.sum().item()
        pixel_weight = self._apply_ohem(ce_loss = ce_loss, pixel_weight=pixel_weight, tot_area=tot_area)

        pixel_loss = pixel_weight * ce_loss
        pixel_loss = pixel_loss.sum()

        pixel_loss /= (1 + NEG_POS_RATIO) * tot_area
        return pixel_loss


if __name__ == "__main__":
    crit = InstanceBalancedCELoss()
    pixel_pred = torch.randn(1, 2, IMG_SIZE, IMG_SIZE)
    pixel_loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
    pixel_loss
