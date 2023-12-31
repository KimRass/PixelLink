import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from einops import rearrange

import config
from model import PixelLink2s
from data import MenuImageDataset
from utils import (
    vis_pixel_pred,
    vis_link_pred,
    vis_gt_bboxes,
    draw_pred_bboxes,
    vis_link_gt,
    vis_pos_pixel_mask,
)
from postprocess import mask_to_bbox


if __name__ == "__main__":
    model = PixelLink2s().to(config.DEVICE)
    state_dict = torch.load(
        "/Users/jongbeomkim/Documents/pixellink_checkpoints/epoch_191.pth",
        map_location=config.DEVICE,
    )
    model.load_state_dict(state_dict["model"])

    ds = MenuImageDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/menu_images",
        img_size=config.IMG_SIZE,
        size_thresh=config.SIZE_THRESH,
        min_area_thresh=config.MIN_AREA_THRESH,
        max_area_thresh=config.MAX_AREA_THRESH,
        split="val",
        stride=config.STRIDE,
    )
    for idx in range(len(ds)):
        image, pixel_gt, link_gt, pixel_weight, bboxes = ds[idx]
        pixel_pred, link_pred = model(image.unsqueeze(0))
        pixel_pred.shape
        all_bboxes = mask_to_bbox(
            pixel_pred=pixel_pred,
            link_pred=link_pred,
            pixel_thresh=0.6,
            link_thresh=0.5,
        )
        draw_pred_bboxes(image, all_bboxes[0])
