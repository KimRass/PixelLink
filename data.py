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
import pandas as pd
import numpy as np
from pathlib import Path
import random
import math
import filetype
from tqdm.auto import tqdm

from utils import _pad_input_image

Image.MAX_IMAGE_PIXELS = None

# np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(edgeitems=4)


def _get_path_pairs(data_dir):
    path_pairs = list()
    for txt_path in Path(data_dir).glob("*.txt"):
        for ext in [".jpg", ".png"]:
            img_path = Path(str(txt_path.with_suffix(ext)).replace("label", "image"))
            if img_path.exists() and filetype.is_image(img_path):
                path_pairs.append((txt_path, img_path))
                break
    return path_pairs


def _get_images(path_pairs):
    images = [Image.open(img_path).convert("RGB") for _, img_path in path_pairs]
    return images


def get_whs(path_pairs):
    images = _get_images(path_pairs)
    whs = [image.size for image in images]
    whs = [(min(w, h), max(w, h)) for w, h in whs]
    return whs


def get_mean_and_std(data_dir):
    path_pairs = _get_path_pairs(data_dir)
    images = _get_images(path_pairs)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for image in tqdm(images):
        tensor = T.ToTensor()(image)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


class MenuImageDataset(Dataset):
    def __init__(self, data_dir, img_size, area_thresh, split="train", mode="2s"):

        self.data_dir = data_dir
        self.img_size = img_size
        self.area_thresh = area_thresh
        self.split = split

        self.scale_factor = 0.5 if mode == "2s" else 0.25
        self.path_pairs = _get_path_pairs(self.data_dir)

        self.color_jitter = T.RandomApply(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
            p=0.5,
            # p=1,
        )
        # image = Image.open("/Users/jongbeomkim/Documents/datasets/menu_images/1_1_image.jpg").convert("RGB")
        # color_jitter(image).show()

    def get_bboxes(self, txt_path):
        bboxes = list()
        with open(txt_path, mode="r") as f:
            for line in f:
                line = line.strip().replace("\ufeff", "")
                splitted = line.split("ᴥ")
                if len(splitted) in [4, 5]:
                    l, t, r, b = splitted[: 4]
                    l = round(float(l.strip()))
                    t = round(float(t.strip()))
                    r = round(float(r.strip()))
                    b = round(float(b.strip()))
                    bboxes.append((l, t, r, b))

        bboxes = pd.DataFrame(bboxes, columns=("l", "t", "r", "b"))
        bboxes["area"] = bboxes.apply(
            lambda x: max(0, (x["r"] - x["l"]) * (x["b"] - x["t"])), axis=1,
        )
        bboxes = bboxes[bboxes["area"] > 0]
        return bboxes

    def _get_textbox_masks(self, bboxes, pos_pixel_mask): # Index 0: Non-text, Index 1: Text
        h, w = pos_pixel_mask.shape
        tboxes = list()
        for row in bboxes.itertuples():
            tbox = torch.zeros((h, w), dtype=torch.bool)
            tbox[row.t: row.b, row.l: row.r] = True
            tbox *= pos_pixel_mask

            tboxes.append(tbox)
        return tboxes

    # "A weight matrix, denoted by $W$, for all positive pixels and selected negative ones."
    def _get_pixel_weight_for_pos_pixels(self, bboxes, pos_pixel_mask):
        def get_areas(tboxes):
            areas = [tbox.sum().item() for tbox in tboxes]
            return areas

        tbox_masks = self._get_textbox_masks(bboxes=bboxes, pos_pixel_mask=pos_pixel_mask)
        n_boxes = len(tbox_masks) # $N$
        if n_boxes == 0:
            pixel_weight = torch.ones(size=(1, self.img_size, self.img_size))
        else:
            areas = get_areas(tbox_masks) # $S_{i}$
            tot_area = sum(areas) # $S = \sum^{N}_{i} S_{i}, \forall i \in {1, \ldots, N}$
            avg_area = tot_area / n_boxes # $B_{i} = S / N$

            weights = list()
            for tbox_mask, area in zip(tbox_masks, areas):
                if area == 0:
                    continue

                weight = avg_area / area # $w_{i} = B_{i} / S_{i}$
                tbox = tbox_mask * weight

                weights.append(tbox)
            if weights:
                pixel_weight = torch.stack(weights, dim=1).sum(dim=1)
                pixel_weight = pixel_weight.unsqueeze(0)
            else:
                pixel_weight = torch.ones(size=(1, self.img_size, self.img_size))
        return pixel_weight

    # "Pixels within text instances are labeled as positive (i.e., text pixels), and otherwise
    # are labeled as negative (i.e., nontext pixels)."
    # Index 0: Non-text, Index 1: Text
    def _get_pos_pixel_mask(self, image: Image.Image, bboxes: pd.DataFrame) -> torch.Tensor:
        w, h = image.size
        canvas = torch.zeros((h, w))
        for row in bboxes.itertuples():
            canvas[row.t: row.b, row.l: row.r] += 1
        # "Pixels inside text bounding boxes are labeled as positive. If overlapping exists,
        # only un-overlapped pixels are positive. Otherwise negative."
        pos_pixel_mask = (canvas == 1)
        return pos_pixel_mask

    def _randomly_scale(self, image, bboxes):
        """
        확대는 하지 않고 축소만 하는데, 가장 작은 바운딩 박스의 넓이가 최소한 `area_thresh`와 같아지는 정도까지만 축소합니다.
        """
        min_area = bboxes["area"].min()
        if math.isnan(min_area):
            minim = 0.25
        else:
            minim = self.area_thresh / min_area
        scale = random.uniform(minim, 1)

        w, h = image.size
        size = (round(h * scale), round(w * scale))
        image = TF.resize(image, size=size, antialias=True)

        bboxes["l"] = bboxes["l"].apply(lambda x: round(x * scale))
        bboxes["t"] = bboxes["t"].apply(lambda x: round(x * scale))
        bboxes["r"] = bboxes["r"].apply(lambda x: round(x * scale))
        bboxes["b"] = bboxes["b"].apply(lambda x: round(x * scale))
        return image, bboxes

    def _randomly_shift_then_crop(self, image, bboxes):
        w, h = image.size
        padding = (max(0, self.img_size - w), max(0, self.img_size - h))
        image = TF.pad(image, padding=padding, padding_mode="constant")
        t, l, h, w = T.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
        image = TF.crop(image, top=t, left=l, height=h, width=w)

        bboxes["l"] += padding[0] - l
        bboxes["t"] += padding[1] - t
        bboxes["r"] += padding[0] - l
        bboxes["b"] += padding[1] - t
        bboxes[["l", "t", "r", "b"]] = bboxes[["l", "t", "r", "b"]].clip(0, self.img_size)
        bboxes = bboxes[(bboxes["l"] != bboxes["r"]) & (bboxes["t"] != bboxes["b"])]
        return image, bboxes

    def _randomly_adjust_b_and_s(self, image):
        image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        return image

    def _get_text_seg_map(self, image: Image.Image, bboxes: pd.DataFrame, pos_pixels):
        w, h = image.size
        canvas = torch.zeros(size=(h, w), dtype=torch.long)
        for idx, row in enumerate(bboxes.itertuples(), start=1):
            canvas[row.t: row.b, row.l: row.r] = idx
        return canvas * pos_pixels

    def _get_pos_links(self, link_seg_map, pos_pixel_mask, stride=5):
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
            shifted = torch.roll(link_seg_map, shifts=shift, dims=(0, 1))
            shifted = (link_seg_map == shifted) * pos_pixel_mask

            ls.append(shifted)
        stacked = torch.stack(ls)
        return stacked

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        txt_path, img_path = self.path_pairs[idx]
        bboxes = self.get_bboxes(txt_path)
        image = Image.open(img_path).convert("RGB")

        if self.split == "train":
            image, bboxes = self._randomly_scale(image=image, bboxes=bboxes)
            image, bboxes = self._randomly_shift_then_crop(image=image, bboxes=bboxes)
            image = self.color_jitter(image)
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

        link_seg_map = self._get_text_seg_map(image=image, bboxes=bboxes, pos_pixels=pos_pixel_mask)
        link_gt = self._get_pos_links(link_seg_map=link_seg_map, pos_pixel_mask=pos_pixel_mask, stride=5)
        link_gt = F.interpolate(
            link_gt.float().unsqueeze(0), scale_factor=self.scale_factor, mode="nearest"
        )[0].long()

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.745, 0.714, 0.681), std=(0.288, 0.300, 0.320))

        data = {
            "image": image,
            "pixel_gt": pixel_gt,
            "link_gt": link_gt,
        }
        if self.split == "train":
            data["pixel_weight"] = pixel_weight
        return data


# if __name__ == "__main__":
#     data_dir = "/Users/jongbeomkim/Documents/datasets/menu_images/"
#     path_pairs = _get_path_pairs(data_dir)
#     len(path_pairs)
#     txt_path, img_path = path_pairs[0]
#     whs

#     ws = sorted([wh[0] for wh in whs])
#     (np.array(ws) > 3000).sum()

#     data_dir = "/Users/jongbeomkim/Documents/datasets/menu_images"
#     ds = MenuImageDataset(data_dir=data_dir, img_size=1024, area_thresh=100)
#     N_WORKERS = 0
#     dl = DataLoader(ds, batch_size=1, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
#     di = iter(dl)
#     for _ in range(len(dl)):
#         data = next(di)
