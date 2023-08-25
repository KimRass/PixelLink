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
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
import random

import config
from utils import draw_bboxes, pos_pixel_mask_to_pil
from infer import _pad_input_image

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)


class MenuImageDataset(Dataset):
    def __init__(self, csv_dir, area_thresh, mode="2s", split="train"):

        self.csv_paths = list(Path(csv_dir).glob("*.csv"))
        self.area_thresh = area_thresh
        self.split = split

        self.scale_factor = 0.5 if mode == "2s" else 0.25

    def get_bboxes(self, csv_path):
        bboxes = pd.read_csv(csv_path, usecols=["xmin", "ymin", "xmax", "ymax"])
        bboxes.rename({"xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2"}, axis=1, inplace=True)
        bboxes["area"] = (bboxes["x2"] - bboxes["x1"]) * (bboxes["y2"] - bboxes["y1"])
        return bboxes

    def load_image(self, csv_path):
        bboxes = pd.read_csv(csv_path, usecols=["image_url"])
        img_path = bboxes["image_url"][0]
        image = Image.open(BytesIO(requests.get(img_path).content)).convert("RGB")
        return image

    def _get_textbox_masks(self, bboxes, pos_pixel_mask): # Index 0: Non-text, Index 1: Text
        h, w = pos_pixel_mask.shape
        tboxes = list()
        for row in bboxes.itertuples():
            tbox = torch.zeros((h, w), dtype=torch.bool)
            tbox[row.y1: row.y2, row.x1: row.x2] = True
            tbox *= pos_pixel_mask

            tboxes.append(tbox)
        return tboxes

    # "A weight matrix, denoted by $W$, for all positive pixels and selected negative ones."
    def _get_pixel_weight_for_pos_pixels(self, bboxes, pos_pixel_mask):
        def get_areas(tboxes):
            areas = [tbox.sum().item() for tbox in tboxes] # $S_{i}$
            return areas

        tbox_masks = self._get_textbox_masks(bboxes=bboxes, pos_pixel_mask=pos_pixel_mask)
        n_boxes = len(tbox_masks) # $N$
        if n_boxes == 0:
            pixel_weight = torch.ones(size=(1, config.IMG_SIZE, config.IMG_SIZE))
        else:
            areas = get_areas(tbox_masks)
            tot_area = sum(areas) # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$
            avg_area = tot_area / n_boxes # $B_{i} = S / N$

            weights = list()
            for tbox_mask, area in zip(tbox_masks, areas):
                if area == 0:
                    continue

                weight = avg_area / area # $w_{i} = B_{i} / S_{i}$
                tbox = tbox_mask * weight

                weights.append(tbox)
            pixel_weight = torch.stack(weights, dim=1).sum(dim=1)
            pixel_weight = pixel_weight.unsqueeze(0)
        return pixel_weight

    # "Pixels within text instances are labeled as positive (i.e., text pixels), and otherwise
    # are labeled as negative (i.e., nontext pixels)."
    def _get_pos_pixel_mask(self, image: Image.Image, bboxes: pd.DataFrame): # Index 0: Non-text, Index 1: Text
        w, h = image.size
        canvas = torch.zeros((h, w))
        for row in bboxes.itertuples():
            canvas[row.y1: row.y2, row.x1: row.x2] += 1
        # "Pixels inside text bounding boxes are labeled as positive. If overlapping exists,
        # only un-overlapped pixels are positive. Otherwise negative."
        pos_pixel_mask = (canvas == 1)
        return pos_pixel_mask

    def _randomly_scale(self, image, bboxes):
        """
        확대는 하지 않고 축소만 하는데, 가장 작은 바운딩 박스의 넓이가 최소한 `area_thresh`와 같아지는 정도까지만 축소합니다.
        """
        min_area = bboxes["area"].min()
        scale = random.uniform(self.area_thresh / min_area, 1)
        
        w, h = image.size
        size = (round(h * scale), round(w * scale))
        image = TF.resize(image, size=size, antialias=True)

        bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * scale))
        bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * scale))
        bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * scale))
        bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * scale))
        return image, bboxes

    def _randomly_shift_then_crop(self, image, bboxes):
        w, h = image.size
        padding = (max(0, config.IMG_SIZE - w), max(0, config.IMG_SIZE - h))
        image = TF.pad(image, padding=padding, padding_mode="constant")
        t, l, h, w = T.RandomCrop.get_params(image, output_size=(config.IMG_SIZE, config.IMG_SIZE))
        image = TF.crop(image, top=t, left=l, height=h, width=w)

        bboxes["x1"] += padding[0] - l
        bboxes["y1"] += padding[1] - t
        bboxes["x2"] += padding[0] - l
        bboxes["y2"] += padding[1] - t
        bboxes[["x1", "y1", "x2", "y2"]] = bboxes[["x1", "y1", "x2", "y2"]].clip(0, config.IMG_SIZE)
        bboxes = bboxes[(bboxes["x1"] != bboxes["x2"]) & (bboxes["y1"] != bboxes["y2"])]
        return image, bboxes

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, idx):
        csv_path = self.csv_paths[idx]

        bboxes = self.get_bboxes(csv_path)
        image = self.load_image(csv_path)

        if self.split == "train":
            # w, h = image.size
            # image = image.resize(size=(w // 4, h // 4), resample=Image.LANCZOS)
            image, bboxes = self._randomly_scale(image=image, bboxes=bboxes)
            image, bboxes = self._randomly_shift_then_crop(image=image, bboxes=bboxes)
        image = _pad_input_image(image)
        # image.show()

        pos_pixel_mask = self._get_pos_pixel_mask(image=image, bboxes=bboxes)

        pixel_gt = pos_pixel_mask.unsqueeze(0)
        pixel_gt = F.interpolate(
            pixel_gt.float().unsqueeze(0), scale_factor=self.scale_factor, mode="nearest"
        )[0].long()

        if self.split == "train":
            pixel_weight = self._get_pixel_weight_for_pos_pixels(
                bboxes=bboxes, pos_pixel_mask=pos_pixel_mask
            )
            pixel_weight = F.interpolate(
                pixel_weight.unsqueeze(0), scale_factor=self.scale_factor, mode="nearest"
            )[0]

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        data = {
            "image": image,
            "pixel_gt": pixel_gt,
        }
        if self.split == "train":
            data["pixel_weight"] = pixel_weight
        return data


if __name__ == "__main__":
    csv_dir = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/data"
    ds = MenuImageDataset(csv_dir=csv_dir)
    N_WORKERS = 0
    dl = DataLoader(ds, batch_size=1, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
    for _ in range(5):
        data = next(iter(dl))

#     data = ds[0]
#     pixel_gt = data["pixel_gt"]
#     pixel_weight = data["pixel_weight"]
#     pixel_gt.shape
#     pixel_weight.shape


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
