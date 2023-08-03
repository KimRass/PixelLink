import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)

# Instance-Balanced Cross-Entropy Loss
# "For a given image with $N$ text instances, all instances are treated equally by giving a same weight
# to everyone of them, denoted as $B_{i}$. For the $i$-th instance with $area = S_{i}$,
# every positive pixels within it have a weight of $w_{i} = B_{i} / S_{i}$."
# "$B_{i} = S / N$"
# "$S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$"

NEG_POS_RATIO = 3 # "$r"


# def get_weight_map(bboxes):
#     n_boxes = len(bboxes) # $N$
#     bboxes["area"] = (bboxes["xmax"] - bboxes["xmin"]) * (bboxes["ymax"] - bboxes["ymin"]) # $S_{i}$
#     tot_area = bboxes["area"].sum() # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$
#     avg_area = tot_area / n_boxes # $B_{i} = S / N$
#     bboxes["weight"] = avg_area / bboxes["area"] # $w_{i} = B_{i} / S_{i}$
#     bboxes.head()


def get_area_list(tboxes):
    areas = [tbox.sum().item() for tbox in tboxes] # $S_{i}$
    return areas

areas = get_area_list(tboxes)
tot_area = sum(areas)

areas = get_area_list(tboxes)
tot_area = 0
for area in areas:


# "A weight matrix, denoted by $W$, for all positive pixels and selected negative ones."
def get_weight_map(w, h, bboxes, pixel_gt):
    tboxes = get_textboxes(w=w, h=h, bboxes=bboxes, pixel_gt=pixel_gt)
    n_boxes = len(tboxes) # $N$

    tot_area = 0
    areas = list()
    for tbox in tboxes:
        area = tbox.sum().item() # $S_{i}$
        tot_area += area # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$

        areas.append(area)
    avg_area = tot_area / n_boxes # $B_{i} = S / N$

    new_tboxes = list()
    for tbox, area in zip(tboxes, areas):
        weight = avg_area / area # $w_{i} = B_{i} / S_{i}$
        # print(weight, end=" ")
        if weight > 4:
            print(weight)
        tbox = tbox.long() * weight

        new_tboxes.append(tbox)
    weight_map = torch.stack(new_tboxes, dim=1).sum(dim=1)
    return weight_map

csv_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471.csv"
img_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471_ori.jpg"

bboxes = get_bboxes(csv_path)
img = load_image(csv_path)
h, w, _ = img.shape

pixel_gt = get_pixel_gt(w=w, h=h, bboxes=bboxes)
pixel_gt = pixel_gt[None, ...].repeat(4, 1, 1)
weight_map = get_weight_map(w=w, h=h, bboxes=bboxes, pixel_gt=pixel_gt)
weight_map.unique()



# "Online Hard Example Mining (OHEM) is applied to select negative pixels. $r * S$ negative pixels
# with the highest losses are selected, by setting their weights to ones. $r$ is the negative-positive ratio
# and is set to 3 as a common practice."

# "The loss on pixel classification task is: Lpixel = 1 (1 + r)S WLpixel CE; (3) where Lpixel CE is the matrix of Cross-Entropy loss on text/non-text prediction. As a result, pixels in small instances have a higher weight, and pixels in large instances have a smaller weight. However, every instance contributes equally to the loss. Loss on Links Losses for positive and negative links"
n_neg_pixels = NEG_POS_RATIO * tot_area

class InstanceBalancedCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        crit = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pixel_pred, pixel_gt):
        pixel_pred = torch.randn(4, 2, 2000, 1500)
        ce_loss = crit(pixel_pred.permute(0, 2, 3, 1).reshape(-1, 2), pixel_gt.view(-1))
        ce_loss.shape