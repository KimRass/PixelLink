# References:
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.ops import box_iou
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

import config
from model import PixelLink2s
from data import MenuImageDataset, _get_path_pairs, get_bboxes
from utils import (
    vis_pixel_pred,
    vis_link_pred,
    vis_gt_bboxes,
    draw_bboxes,
    vis_link_gt,
    vis_pos_pixel_mask,
    _pad_input_image,
)
from postprocess import mask_to_bbox


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args



def bboxes_to_str(bboxes):
    return [
        str(bbox)[1: -1] + "\n" if idx != len(bboxes) -1 else str(bbox)[1: -1]
        for idx, bbox in enumerate(bboxes)
    ]


def infer_using_every_image(model, data_dir, save_dir):
    path_pairs = _get_path_pairs(data_dir)
    save_dir = Path(save_dir)
    for txt_path, img_path in tqdm(path_pairs):
        save_path = save_dir/txt_path.name
        if not save_path.exists():
            image = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(image)
            img_tensor = TF.normalize(img_tensor, mean=config.IMG_MEAN, std=config.IMG_STD)
            img_tensor = _pad_input_image(img_tensor.unsqueeze(0))
            pixel_pred, link_pred = model(img_tensor)

            pred_bboxes = mask_to_bbox(
                pixel_pred=pixel_pred[0],
                link_pred=link_pred[0],
                pixel_thresh=0.6,
                link_thresh=0.5,
            )
            draw_bboxes(image=image, pred_bboxes=pred_bboxes)
            with open(save_dir/txt_path.name, mode="w") as f:
                f.writelines(bboxes_to_str(pred_bboxes))


def get_gt_bboxes(txt_path):
    gt_boxes = list()
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
                gt_boxes.append((l, t, r, b))
    return gt_boxes


def get_pred_bboxes(txt_path):
    pred_bboxes = list()
    with open(txt_path, mode="r") as f:
        for line in f:
            pred_bboxes.append(tuple(map(int, line.strip().split(", "))))
    return pred_bboxes


def get_iou(bbox1, bbox2):
    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2

    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)

    l_inter = max(l1, l2)
    r_inter = min(r1, r2)
    t_inter = max(t1, t2)
    b_inter = min(b1, b2)
    
    area_inter = max(0, r_inter - l_inter) * max(0, b_inter - t_inter)
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    return iou


if __name__ == "__main__":
    args = get_args()

    model = PixelLink2s().to(config.DEVICE)
    state_dict = torch.load(
        "/Users/jongbeomkim/Documents/pixellink_checkpoints/epoch_191.pth",
        map_location=config.DEVICE,
    )
    model.load_state_dict(state_dict["model"])

    infer_using_every_image(model=model, data_dir=args.data_dir, save_dir=args.save_dir)


    # data_dir = "/Users/jongbeomkim/Documents/datasets/menu_images"
    # save_dir = "/Users/jongbeomkim/Documents/pixellink_evaluation"
    save_dir = Path(save_dir)
    path_pairs = _get_path_pairs(data_dir)
    for gt_path, img_path in tqdm(path_pairs):
        gt_bboxes = get_gt_bboxes(gt_path)
        pred_path = save_dir/gt_path.name
        if pred_path.exists():
            pred_bboxes = get_pred_bboxes(pred_path)

            # image = Image.open(img_path).convert("RGB")
            # draw_bboxes(image=image, bboxes=pred_bboxes)
            # draw_bboxes(image=image, bboxes=gt_bboxes)

            gt_indices = list()
            pred_indices = list()
            ious = list()
            for pred_idx, pred_bbox in enumerate(pred_bboxes):
                for gt_idx, gt_bbox in enumerate(gt_bboxes):
                    iou = get_iou(pred_bbox[1:], gt_bbox)
                    if iou >= 0.5:
                        # ls.append((iou, pred_idx, gt_idx))
                        pred_indices
                        gt_indices
            ls
            np.argsort([ils)
