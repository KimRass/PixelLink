import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import config
from model import PixelLink2s
from data import MenuImageDataset
from utils import vis_pixel_pred, draw_bboxes
from postprocess import mask_to_bbox


if __name__ == "__main__":
    model = PixelLink2s().to(config.DEVICE)
    state_dict = torch.load("/Users/jongbeomkim/Documents/epoch_113.pth", map_location=config.DEVICE)
    model.load_state_dict(state_dict["model"])

    ds = MenuImageDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/menu_images",
        img_size=config.IMG_SIZE,
        size_thresh=config.SIZE_THRESH,
        min_area_thresh=config.MIN_AREA_THRESH,
        max_area_thresh=config.MAX_AREA_THRESH,
        split="val",
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    for image, pixel_gt, link_gt, pixel_weight in dl:
        pixel_pred, link_pred = model(image)
        vis_pixel_pred(image[0], pixel_pred[0])

        image.shape
        all_bboxes = mask_to_bbox(
            pixel_pred=pixel_pred,
            link_pred=link_pred,
            pixel_thresh=0.8,
            link_thresh=0.8,
        )
        draw_bboxes(image[0], all_bboxes[0])
