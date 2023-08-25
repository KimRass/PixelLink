import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from time import time
from datetime import timedelta

import config


def _pad_input_image(image):
    """
    Resize the image so that the width and the height are multiples of 16 each.
    """
    # _, _, h, w = image.shape
    w, h = image.size
    if h % 16 != 0:
        new_h = h + (16 - h % 16)
    else:
        new_h = h
    if w % 16 != 0:
        new_w = w + (16 - w % 16)
    else:
        new_w = w
    new_image = TF.pad(image, padding=(0, 0, new_w - w, new_h - h), padding_mode="constant")
    return new_image


def draw_bboxes(image: Image.Image, bboxes: pd.DataFrame) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for row in bboxes.itertuples(): # Draw bboxes
        draw.rectangle(
            xy=(row.x1, row.y1, row.x2, row.y2),
            outline="rgb(255, 0, 0)",
            fill=None,
            width=2,
        )
        draw.line(xy=(row.x1, row.y1, row.x2, row.y2), fill="rgb(255, 0, 0)", width=1)
        draw.line(xy=(row.x1, row.y2, row.x2, row.y1), fill="rgb(255, 0, 0)", width=1)
    return canvas


def pos_pixel_mask_to_pil(pos_pixel_mask):
    return Image.fromarray((pos_pixel_mask.numpy().astype("uint8") * 255))


if __name__ == "__main__":
    # image.min(), image.max()
    grid = make_grid(val_image, normalize=True, value_range=(-1, 1))
    TF.to_pil_image(grid).show()

    grid = make_grid(val_pixel_gt.float())
    TF.to_pil_image(grid).show()

    val_pixel_gt.shape, val_pixel_pred.shape

    val_pixel_pred.sum(dim=1)
    # grid = make_grid((val_pixel_pred[:, 1, ...] >= 0.5).float())
    grid = make_grid((val_pixel_pred[:, 0, ...] >= 0.5).float())
    # grid.min(), grid.max()
    TF.to_pil_image(grid).show()


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def split(pixel_pred):
    temp = F.interpolate(pixel_pred, scale_factor=2, mode="nearest")
    temp = temp[:, 1, ...]
    # temp = (temp[:, 1, ...] >= 0.5).float()
    return temp


def vis(image, pixel_pred):
    temp = split(pixel_pred)
    temp = TF.to_pil_image(temp)
    temp.show()
    pil_image = TF.to_pil_image((image * 0.5) + 0.5)

    Image.blend(pil_image, temp, alpha=0.5)


def _to_pil(img, mode="RGB"):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img, mode=mode)
        return image
    else:
        return img


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def postprocess_pixel_pred(pixel_pred):
    pixel_pred = pixel_pred.detach().cpu().numpy()
    pixel_pred = pixel_pred[0, 1, ...]
    h, w = pixel_pred.shape
    pixel_pred = cv2.resize(pixel_pred, dsize=(w * 2, h * 2))
    pixel_pred *= 255
    pixel_pred = pixel_pred.astype("uint8")
    pixel_pred = _apply_jet_colormap(pixel_pred)
    return pixel_pred


def vis_pixel_pred(image, pixel_pred, alpha):
    pixel_pred = postprocess_pixel_pred(pixel_pred)
    pil_image = TF.to_pil_image((image * 0.5) + 0.5)
    pil_pixel_pred = _to_pil(pixel_pred)
    blended = Image.blend(pil_image, pil_pixel_pred, alpha=alpha)
    blended.show()


# def vis_link_gt(link_gt):


def _get_canvas_same_size_as_image(img, black=False):
    if black:
        return np.zeros_like(img).astype("uint8")
    else:
        return (np.ones_like(img) * 255).astype("uint8")


def _repaint_segmentation_map(seg_map):
    canvas_r = _get_canvas_same_size_as_image(seg_map, black=True)
    canvas_g = _get_canvas_same_size_as_image(seg_map, black=True)
    canvas_b = _get_canvas_same_size_as_image(seg_map, black=True)

    remainder_map = seg_map % len(config.COLORS) + 1
    for remainder, (r, g, b) in enumerate(config.COLORS, start=1):
        canvas_r[remainder_map == remainder] = r
        canvas_g[remainder_map == remainder] = g
        canvas_b[remainder_map == remainder] = b
    canvas_r[seg_map == 0] = 0
    canvas_g[seg_map == 0] = 0
    canvas_b[seg_map == 0] = 0

    dstacked = np.dstack([canvas_r, canvas_g, canvas_b])
    return dstacked


def segment_pixel_pred(pixel_pred):
    pixel_pred = pixel_pred.detach().cpu().numpy()
    pixel_pred = pixel_pred[0, 1, ...]
    h, w = pixel_pred.shape
    pixel_pred = cv2.resize(pixel_pred, dsize=(w * 2, h * 2))
    pixel_pred = (pixel_pred >= 0.5)
    _, seg_map = cv2.connectedComponents(image=pixel_pred.astype("uint8"), connectivity=4)
    return seg_map
