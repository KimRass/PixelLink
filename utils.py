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

COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 255, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 250),
    (240, 50, 230),
    (210, 255, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
)


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_3d(img):
    if img.ndim == 2:
        return np.dstack([img, img, img])
    else:
        return img


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

    Image.blend(pil_image, temp, alpha=0.6)


def _to_pil(img, mode="RGB"):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img, mode=mode)
        return image
    else:
        return img


def _to_array(img):
    img = np.array(img)
    return img


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _get_w_and_h(img):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    return w, h


def _resize_image(img, w, h):
    ori_w, ori_h = _get_w_and_h(img)
    if w < ori_w or h < ori_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    resized = cv2.resize(src=img, dsize=(w, h), interpolation=interpolation)
    return resized


def resize_with_thresh(image, size_thresh=3000):
    w, h = image.size
    if min(w, h) > size_thresh:
        if w < h:
            ori_w = round(size_thresh / w * w)
            ori_h = round(size_thresh / w * h)
        else:
            ori_w = round(size_thresh / h * w)
            ori_h = round(size_thresh / h * h)
        new_image = image.resize(size=(ori_w, ori_h))
        return new_image
    else:
        return image


def _get_canvas_same_size_as_image(img, black=False):
    if black:
        return np.zeros_like(img).astype("uint8")
    else:
        return (np.ones_like(img) * 255).astype("uint8")


def draw_bboxes(bboxes, image):
    canvas = _to_pil(_get_canvas_same_size_as_image(_to_array(image), black=True))
    draw = ImageDraw.Draw(canvas)
    dic = dict()
    for row in bboxes.itertuples():
        h = row.b - row.t
        w = row.r - row.l
        smaller = min(w, h)
        thickness = max(1, smaller // 22)

        dic[row.Index] = ((0, 255, 0), (0, 100, 0), thickness)

    for row in bboxes.itertuples():
        _, fill, thickness = dic[row.Index]
        draw.rectangle(
            xy=(row.l, row.t, row.r, row.b),
            outline=None,
            fill=fill,
            width=thickness
        )
    for row in bboxes.itertuples():
        outline, _, thickness = dic[row.Index]
        draw.rectangle(
            xy=(row.l, row.t, row.r, row.b),
            outline=outline,
            fill=None,
            width=thickness
        )

    # if index:
    #     max_len = max(map(len, map(str, bboxes.index)))
    #     for row in bboxes.itertuples():
    #         h = row.b - row.t
    #         w = row.r - row.l
    #         smaller = min(w, h)
    #         font_size = max(10, min(40, smaller // 4))

    #         draw.text(
    #             xy=(row.l, row.t - 4),
    #             text=str(row.Index).zfill(max_len),
    #             fill="white",
    #             stroke_fill="black",
    #             stroke_width=2,
    #             font=ImageFont.truetype(font=font_path, size=int(font_size)),
    #             anchor="ls"
    #         )
    Image.blend(_to_pil(canvas), im2=image, alpha=0.5).show()


def postprocess_pixel_gt(pixel_gt):
    copied = pixel_gt.clone()
    copied = copied.detach().cpu().numpy()
    copied = copied[0, ...]
    h, w = copied.shape
    copied *= 255
    copied = copied.astype("uint8")
    copied = cv2.resize(copied, dsize=(w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    copied = _apply_jet_colormap(copied)
    return copied


def postprocess_link_pred(link_pred):
    link_pred = link_pred.detach().cpu().numpy()
    link_pred = link_pred[0, 8, ...]
    h, w = link_pred.shape
    link_pred *= 255
    link_pred = link_pred.astype("uint8")
    link_pred = cv2.resize(link_pred, dsize=(w * 2, h * 2))
    link_pred = _apply_jet_colormap(link_pred)
    return link_pred


def _denorm_image(image, mean=(0.745, 0.714, 0.681), std=(0.288, 0.300, 0.320)):
    copied = image.clone()
    copied *= torch.as_tensor(std)[:, None, None]
    copied += torch.as_tensor(mean)[:, None, None]
    copied = TF.to_pil_image(copied)
    return copied


def vis_pixel_gt(image, pixel_gt, alpha=0.6):
    pil_image = _denorm_image(image)
    pixel_gt = postprocess_pixel_gt(pixel_gt)
    pil_pixel_gt = _to_pil(pixel_gt)
    blended = Image.blend(pil_image, pil_pixel_gt, alpha=alpha)
    blended.show()


def postprocess_pixel_pred(pixel_pred):
    copied = pixel_pred[1, ...].clone()
    copied = copied.detach().cpu().numpy()
    h, w = copied.shape
    copied *= 255
    copied = copied.astype("uint8")
    copied = cv2.resize(copied, dsize=(w * 2, h * 2))
    copied = _apply_jet_colormap(copied)
    return _to_pil(copied)


def vis_pixel_pred(image, pixel_pred, alpha=0.6):
    pil_image = _denorm_image(image)
    pil_pixel_pred = postprocess_pixel_pred(pixel_pred)
    blended = Image.blend(pil_image, pil_pixel_pred, alpha=alpha)
    blended.show()


def vis_link_pred(image, link_pred, alpha=0.6):
    link_pred = postprocess_link_pred(link_pred)
    pil_image = TF.to_pil_image((image * 0.5) + 0.5)
    pil_link_pred = _to_pil(link_pred)
    blended = Image.blend(pil_image, pil_link_pred, alpha=alpha)
    blended.show()


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


def draw_bboxes(image, bboxes):
    # bboxes = all_bboxes[0]
    pil_image = _denorm_image(image)
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        # draw.polygon(xy=[tuple(i) for i in quad.tolist()], outline=(255, 0, 0))
        draw.rectangle(xy=bbox, outline="red", width=2)
    pil_image.show()


# def draw_quads(image, quads):
#     quads = all_bboxes[0]
#     pil_image = _denorm_image(image[0])
#     draw = ImageDraw.Draw(pil_image)
#     for quad in quads:
#         draw.polygon(xy=[tuple(i) for i in quad.tolist()], outline=(255, 0, 0))
#     pil_image.show()


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
