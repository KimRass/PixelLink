import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

import config

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)


class InstanceBalancedCELoss(nn.Module):
    # def __init__(self, lamb=2):
    def __init__(self):
        super().__init__()

        # self.lamb = lamb

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def _apply_ohem(self, ce_loss, pixel_weight, tot_area):
        temp_loss = ce_loss.clone()
        temp_loss[pixel_weight != 0] = 0
        sorted_indices = torch.sort(temp_loss, descending=True)[1]
        n_neg_pixels = config.NEG_POS_RATIO * tot_area
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
    def forward(self, pixel_pred, pixel_gt, pixel_weight, link_pred, link_gt):
        copied = pixel_weight.clone()

        ### Pixel loss
        pixel_pred = rearrange(pixel_pred, pattern="b c h w -> (b h w) c")
        pixel_gt = rearrange(pixel_gt, pattern="b c h w -> (b h w) c").squeeze()
        pixel_weight = rearrange(pixel_weight, pattern="b c h w -> (b h w) c").squeeze()

        # "The loss on pixel classification task is: $L_{pixel} = \frac{1}{1 + r)S}WL_{pixel_CE}$
        # where $L_{pixel_CE} is the matrix of Cross-Entropy loss on text/non-text prediction.
        # As a result, pixels in small instances have a higher weight, and pixels in large instances
        # have a smaller weight. However, every instance contributes equally to the loss."
        pixel_ce_loss = self.ce(pixel_pred, pixel_gt)
        tot_area = pixel_gt.sum().item()
        pixel_weight = self._apply_ohem(
            ce_loss=pixel_ce_loss, pixel_weight=pixel_weight, tot_area=tot_area,
        )

        pixel_loss = (pixel_weight * pixel_ce_loss).sum()
        pixel_loss /= (1 + config.NEG_POS_RATIO) * tot_area

        ### Link loss
        pixel_weight = copied()
        # "$W_{pos_link}(i, j, k) = W(i, j) * (Y_link(i, j, k) == 1)$"
        pos_link_weight = (pixel_weight * (link_gt == 1))
        # "$W_{neg_link}(i, j, k) = W(i, j) * (Y_link(i, j, k) == 0)$"
        neg_link_weight = (pixel_weight * (link_gt == 0))

        link_pred = rearrange(link_pred, pattern="b (m n) h w -> (b h w n) m", m=2)
        link_gt = rearrange(link_gt, pattern="b c h w -> (b h w c)")

        link_ce_loss = self.ce(link_pred, link_gt) # "$L_{link_CE}$"
        # "$L_{link_pos} = W_{pos_link}L_{link_CE}$"
        pos_link_ce_loss = pos_link_weight.view(-1) * link_ce_loss
        # "$L_{link_neg} = W_{neg_link}L_{link_CE}$"
        neg_link_ce_loss = neg_link_weight.view(-1) * link_ce_loss

        link_loss = (pos_link_ce_loss / pos_link_weight.sum()) + (neg_link_ce_loss / neg_link_weight.sum())
        link_loss = link_loss.sum()

        # loss = self.lamb * pixel_loss + link_loss
        # return loss
        return pixel_loss, link_loss


if __name__ == "__main__":
    crit = InstanceBalancedCELoss()
    pixel_pred = torch.randn(1, 2, config.IMG_SIZE, config.IMG_SIZE)
    pixel_loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
    pixel_loss
