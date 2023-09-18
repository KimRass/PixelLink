import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from einops import rearrange

import config
from model import PixelLink2s
from data import MenuImageDataset
from utils import vis_pixel_pred, vis_link_pred, vis_gt_bboxes, draw_pred_bboxes, vis_link_gt
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
    )
    # dl = DataLoader(
    #     ds,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # di = iter(dl)

    # for image, pixel_gt, link_gt, pixel_weight in dl:
    # image, pixel_gt, link_gt, pixel_weight = next(di)
    for idx in range(len(ds)):
        idx = 7
        image, pixel_gt, link_gt, pixel_weight, bboxes = ds[idx]
        image = image[:, 1300: 1428, :]
        pixel_gt = pixel_gt[:, 650: 714, :]
        link_gt = link_gt[:, 650: 714, :]

        pixel_pred, link_pred = model(image.unsqueeze(0))

        # vis_gt_bboxes(image=image, bboxes=bboxes)

        vis_pixel_pred(image, pixel_pred[0])
        # vis_link_pred(image, link_pred[0])
        vis_link_gt(image, link_gt)

        # all_bboxes = mask_to_bbox(
        #     pixel_pred=pixel_pred,
        #     link_pred=link_pred,
        #     pixel_thresh=0.8,
        #     link_thresh=0.9,
        # )
        # draw_pred_bboxes(image, all_bboxes[0])

        pixel_pred[0, 1, ...] = pixel_gt[0]
        for idx in range(8):
            link_pred[0, 8 + idx, ...] = link_gt[idx]
        all_bboxes = mask_to_bbox(
            pixel_pred=pixel_pred,
            link_pred=link_pred,
            pixel_thresh=0.8,
            link_thresh=0.9,
        )
        draw_pred_bboxes(image, all_bboxes[0])
