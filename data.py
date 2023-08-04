# "For a given pixel and one of its eight neighbors, if they belong to the same instance,
# the link between them is positive. Otherwise negative. Note that ground truth calculation
# is carried out on input images resized to the shape of prediction layer, i.e., 'conv3_3' for '4s'
# and 'conv2_2' for '2s'."

# "Given predictions on pixels and links, two different thresholds can be applied on them separately.
# Positive pixels are then grouped together using positive links, resulting in a collection of CCs,
# each representing a detected text instance. Given two neighboring positive pixels, their link
# are predicted by both of them, and they should be connected when one or both of the two link predictions
# are positive. This linking process can be implemented using disjoint-set data structure."

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from tqdm.auto import tqdm
from pathlib import Path
import random

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)

IMG_SIZE = 512


def get_bboxes(csv_path):
    bboxes = pd.read_csv(csv_path, usecols=["xmin", "ymin", "xmax", "ymax"])
    bboxes.rename({"xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2"}, axis=1, inplace=True)
    bboxes["area"] = (bboxes["x2"] - bboxes["x1"]) * (bboxes["y2"] - bboxes["y1"])
    # for col in bboxes.columns.tolist():
    #     bboxes[col] = bboxes[col].apply(lambda x: round(x / 2))
    return bboxes


def load_image(csv_path):
    bboxes = pd.read_csv(csv_path, usecols=["image_url"])
    img_path = bboxes["image_url"][0]
    image = Image.open(BytesIO(requests.get(img_path).content)).convert("RGB")
    return image
    # w, h = image.size
    # image = image.resize(size=(w // 2, h // 2))
    # img = np.array(image)
    # return img


# def get_text_seg_map(w, h, pixel_gt):
#     # canvas = np.zeros((h, w), dtype="uint16")
#     canvas = np.zeros((h, w), dtype="uint8")
#     for idx, row in enumerate(bboxes.itertuples(), start=1):
#         canvas[row.y1: row.y2, row.1: row.x2] = idx
#     return canvas * pixel_gt


def get_pos_links(seg_map, stride=5):
    ls = list()
    for shift in [
        (0, stride), # "Left"
        (-stride, stride), # "Left-down"
        (stride, stride), # "Left-up"
        (0, -stride), # "Right"
        (-stride, -stride), # "Right-down"
        (stride, -stride), # "Right-up"
        (stride, 0), # "Up"
        (-stride, 0), # "Down"
    ]:
        shifted = np.roll(seg_map, shift=shift, axis=(0, 1))
        shifted = (seg_map == shifted) * pixel_gt

        ls.append(shifted)
    stacked = np.stack(ls)
    return stacked


def get_areas(tboxes):
    areas = [tbox.sum().item() for tbox in tboxes] # $S_{i}$
    return areas


class MenuImageDataset(Dataset):
    def __init__(self, csv_dir):

        self.csv_paths = list(Path(csv_dir).glob("*.csv"))

    def _get_textbox_masks(self, w, h, bboxes, pos_pixel_mask): # Index 0: Non-text, Index 1: Text
        tboxes = list()
        for row in bboxes.itertuples():
            tbox = torch.zeros((h, w), dtype=torch.bool)
            tbox[row.y1: row.y2, row.x1: row.x2] = True
            tbox = tbox * pos_pixel_mask

            tboxes.append(tbox)
        return tboxes

    # "A weight matrix, denoted by $W$, for all positive pixels and selected negative ones."
    def _get_pixel_weight_mat_for_pos_pixels(self, w, h, bboxes, pos_pixel_mask):
        tbox_masks = self._get_textbox_masks(w=w, h=h, bboxes=bboxes, pos_pixel_mask=pos_pixel_mask)
        n_boxes = len(tbox_masks) # $N$

        areas = get_areas(tbox_masks)
        tot_area = sum(areas) # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$
        avg_area = tot_area / n_boxes # $B_{i} = S / N$

        weights = list()
        for tbox_mask, area in zip(tbox_masks, areas):
            weight = avg_area / area # $w_{i} = B_{i} / S_{i}$
            tbox = tbox_mask * weight

            weights.append(tbox)
        weight_mat = torch.stack(weights, dim=1).sum(dim=1)
        return weight_mat

    # "Pixels within text instances are labeled as positive (i.e., text pixels), and otherwise
    # are labeled as negative (i.e., nontext pixels)."
    def _get_pos_pixel_mask(self, w, h, bboxes): # Index 0: Non-text, Index 1: Text
        canvas = np.zeros((h, w), dtype="uint8")
        for row in bboxes.itertuples():
            canvas[row.y1: row.y2, row.x1: row.x2] += 1
        # "Pixels inside text bounding boxes are labeled as positive. If overlapping exists,
        # only un-overlapped pixels are positive. Otherwise negative."
        pos_pixel_mask = (canvas == 1)
        # pos_pixel_mask = torch.Tensor(pos_pixel_mask).long()
        pos_pixel_mask = torch.tensor(pos_pixel_mask).unsqueeze(0)
        return pos_pixel_mask

    def _randomly_scale(self, image, pos_pixel_mask, area_thresh=1500):
        w, h = image.size
        scale = random.uniform(area_thresh / self.min_area, 1)
        
        size = (round(h * scale), round(w * scale))
        image = TF.resize(image, size=size, antialias=True)
        pos_pixel_mask = TF.resize(pos_pixel_mask, size=size, antialias=True)
        return image, pos_pixel_mask

        # bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * scale))
        # bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * scale))
        # bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * scale))
        # bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * scale))

    def _randomly_shift_then_crop(self, image, pos_pixel_mask):
        w, h = image.size
        padding = (max(0, IMG_SIZE - w), max(0, IMG_SIZE - h))
        image = TF.pad(image, padding=padding, padding_mode="constant")
        t, l, h, w = T.RandomCrop.get_params(image, output_size=(IMG_SIZE, IMG_SIZE))
        image = TF.crop(image, top=t, left=l, height=h, width=w)

        pos_pixel_mask = TF.pad(pos_pixel_mask, padding=padding, padding_mode="constant")
        pos_pixel_mask = TF.crop(pos_pixel_mask, top=t, left=l, height=h, width=w)
        return image, pos_pixel_mask

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, idx):
        csv_path = self.csv_paths[idx]

        bboxes = get_bboxes(csv_path)
        image = load_image(csv_path)

        self.min_area = bboxes["area"].min()

        w, h = image.size
        pos_pixel_mask = self._get_pos_pixel_mask(w=w, h=h, bboxes=bboxes)
        image, pos_pixel_mask = self._randomly_scale(image=image, pos_pixel_mask=pos_pixel_mask)
        image, pos_pixel_mask = self._randomly_shift_then_crop(image=image, pos_pixel_mask=pos_pixel_mask)

        pixel_gt = torch.stack([~pos_pixel_mask, pos_pixel_mask]).long()
        pixel_weight = self._get_pixel_weight_mat_for_pos_pixels(
            w=w, h=h, bboxes=bboxes, pos_pixel_mask=pos_pixel_mask
        ).unsqueeze(0)
        return {
            "image": image,
            "pos_pixel_mask": pos_pixel_mask,
            "pixel_gt": pixel_gt,
            "pixel_weight": pixel_weight,
        }
        # return image, pos_pixel_mask


csv_dir = "/Users/jongbeomkim/Desktop/workspace/text_segmenter"
ds = MenuImageDataset(csv_dir=csv_dir)
for _ in range(10):
    image, pos_pixel_mask, pixel_gt, pixel_weight = ds[0]
    show_image(image, (pos_pixel_mask[0].numpy() * 255).astype("uint8"))

data = ds[0]
pixel_gt = data["pixel_gt"]
pixel_weight = data["pixel_weight"]
pixel_gt.shape
pixel_weight.shape

# dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
# data = next(iter(dl))




# seg_map = get_text_seg_map(w=w, h=h, pixel_gt=pixel_gt)
# pos_links = get_pos_links(seg_map=seg_map, stride=1)
# temp = (sum(pos_links) == 8)
# show_image(img, temp)

# # show_image(seg_map, pos_links[7])
# show_image(img, pos_links[7])


# _, out = cv2.connectedComponents(image=temp.astype("uint8"), connectivity=4)
# show_image(out)




# # cls_logits = torch.stack((torch.Tensor(~pixel_gt), torch.Tensor(pixel_gt)))[None, ...].repeat(2, 1, 1, 1)
# # link_logits = torch.cat(
# #     (torch.stack([torch.Tensor(~i) for i in pos_links]), torch.stack([torch.Tensor(i) for i in pos_links])),
# #     dim=0
# # )[None, ...].repeat(2, 1, 1, 1)
# pixel_pos_scores = torch.Tensor(pixel_gt)[None, ...].repeat(2, 1, 1)
# link_pos_scores = torch.stack([torch.Tensor(i) for i in pos_links])[None, ...].repeat(2, 1, 1, 1)
# # mask, bboxes = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
# mask = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
# # np.unique(mask)
# show_image(mask)
# show_image(img, mask)
# save_image(mask, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output.png")
# save_image(mask, img, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output2.png")
