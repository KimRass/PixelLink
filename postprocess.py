import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn._get_segtional as F
from PIL import ImageDraw

import config
from utils import _get_canvas_same_size_as_image, _to_pil, _to_3d


def get_neighbors(h_idx, w_idx):
    neighbors = list()
    neighbors.append((h_idx, w_idx - 1)) # "Left"
    neighbors.append((h_idx + 1, w_idx - 1)) # "Left-down"
    neighbors.append((h_idx - 1, w_idx - 1)) # "Left-up"
    neighbors.append((h_idx, w_idx + 1)) # "Right"
    neighbors.append((h_idx + 1, w_idx + 1)) # "Right-down"
    neighbors.append((h_idx - 1, w_idx + 1)) # "Right-up"
    neighbors.append((h_idx + 1, w_idx)) # "Up"
    neighbors.append((h_idx - 1, w_idx)) # "Down"
    # neighbors.append((h_idx - 1, w_idx - 1)) # "Left-up"
    # neighbors.append((h_idx + 1, w_idx)) # "Up"
    # neighbors.append((h_idx, w_idx - 1)) # "Left"
    # neighbors.append((h_idx - 1, w_idx + 1)) # "Right-up"
    # neighbors.append((h_idx, w_idx + 1)) # "Right"
    # neighbors.append((h_idx + 1, w_idx + 1)) # "Right-down"
    # neighbors.append((h_idx - 1, w_idx)) # "Down"
    # neighbors.append((h_idx + 1, w_idx - 1)) # "Left-down"
    return neighbors


def _find(x, group_mask):
    while group_mask.get(x) != -1:
        x = group_mask.get(x)
    return x


def _union(x, y, group_mask):
    roota = _find(x, group_mask=group_mask)
    rootb = _find(y, group_mask=group_mask)
    if roota != rootb:
        group_mask[rootb] = roota
    return


def _get_seg(pixel_cls, link_cls):
    # pixel_cls = pixel_pred_temp[0]
    # link_cls = link_neighbors[0]
    pixel_cls = pixel_cls.detach().cpu().numpy()
    link_cls = link_cls.detach().cpu().numpy()

    pixel_points = list(zip(*np.where(pixel_cls)))
    group_mask = dict.fromkeys(pixel_points, -1)
    # len(pixel_points)

    h, w = pixel_cls.shape
    for h_idx, w_idx in pixel_points:
        neighbors = get_neighbors(h_idx, w_idx)
        for i, (nh_idx, nw_idx) in enumerate(neighbors):
            if (nh_idx < 0) or (nw_idx < 0) or (nh_idx >= h) or (nw_idx >= w):
                continue
            if (pixel_cls[nh_idx, nw_idx] == 1) and (link_cls[i, h_idx, w_idx] == 1):
                _union(x=(h_idx, w_idx), y=(nh_idx, nw_idx), group_mask=group_mask)

    out = np.zeros(pixel_cls.shape, dtype="int32")
    root_map = dict()
    for h_idx, w_idx in pixel_points:
        root = _find((h_idx, w_idx), group_mask=group_mask)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        out[h_idx, w_idx] = root_map[root]
    return out


def _get_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _filter_short_side(bbox):
    for i, point in enumerate(bbox):
        if _get_dist(point, bbox[(i + 1) % 4]) < 5 ** 2:
            return True
    return False


def _quad_to_rect(quad):
    l = quad[:, 0].min()
    t = quad[:, 1].min()
    r = quad[:, 0].max()
    b = quad[:, 1].max()
    return (l, t, r, b)


def mask_to_bbox(
    pixel_pred, link_pred, mode="2s", pixel_thresh=0.5, link_thresh=0.5, area_thresh=100,
):
    """
    pixel_pred: batch_size * 2 * H * W
    link_pred: batch_size * 16 * H * W
    """
    # mode="2s"
    # area_thresh=100
    batch_size, _, mask_height, mask_width = pixel_pred.shape

    pixel_pred_temp = pixel_pred[:, 1] > pixel_thresh
    link_pred_temp = link_pred[:, 8:] > link_thresh

    link_neighbors = torch.zeros(
        [batch_size, config.N_NEIGHBORS, mask_height, mask_width],
        dtype=torch.uint8,
        device=pixel_pred_temp.device,
    )
    for i in range(config.N_NEIGHBORS):
        link_neighbors[:, i] = link_pred_temp[:, i] & pixel_pred_temp

    all_bboxes = list()
    for i in range(batch_size):
        seg = _get_seg(pixel_pred_temp[i], link_neighbors[i])
        _to_pil(_repaint_segmentation_map(seg)).show()
        n_bboxes = np.max(seg)

        bboxes = list()
        for i in range(1, n_bboxes + 1):
            box_mask = (seg == i).astype("uint8")
            if box_mask.sum() < area_thresh:
                continue

            # _to_pil(_to_3d((box_mask * 255))).show()
            contours, _ = cv2.findContours(
                box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE,
            )
            quad = cv2.minAreaRect(contours[0])
            quad = cv2.boxPoints(quad)
            # if _filter_short_side(quad):
            #     continue

            scale = 2 if mode == "2s" else 4
            quad = (quad * scale).astype("uint16")
            rect = _quad_to_rect(quad)
            bboxes.append(rect)
        all_bboxes.append(bboxes)
    return all_bboxes


if __name__ == "__main__":
    out = mask_to_bbox(pixel_pred, link_pred, 8, 2)
    out
