import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import config
from utils import _get_canvas_same_size_as_image, _to_pil, _to_3d, _repaint_segmentation_map


def get_neighbors(center):
    return (
        (center[0], center[1] - 1), # "Left"
        (center[0] - 1, center[1] - 1), # "Left-up"
        (center[0] + 1, center[1] - 1), # "Left-down"
        (center[0], center[1] + 1), # "Right"
        (center[0] - 1, center[1] + 1), # "Right-up"
        (center[0] + 1, center[1] + 1), # "Right-down"
        (center[0] + 1, center[1]), # "Down"
        (center[0] - 1, center[1]), # "Up"
        # (center[0] - 1, center[1] - 1), # "Left-up"
        # (center[0] + 1, center[1]), # "Down"
        # (center[0], center[1] - 1), # "Left"
        # (center[0] - 1, center[1] + 1), # "Right-up"
        # (center[0], center[1] + 1), # "Right"
        # (center[0] + 1, center[1] + 1), # "Right-down"
        # (center[0] - 1, center[1]), # "Up"
        # (center[0] + 1, center[1] - 1), # "Left-down"
    )


def _find(x, parent):
    # while parent.get(x) != -1:
    #     x = parent.get(x)
    # return x
    if parent[x] == -1:
        return x
    else:
        parent[x] = _find(parent[x], parent=parent)
        return parent[x]


def _union(x, y, parent):
    x_rep = _find(x, parent=parent)
    y_rep = _find(y, parent=parent)
    if x_rep != y_rep:
        parent[y_rep] = x_rep
    # x_rep = _find(x, parent=parent)
    # y_rep = _find(y, parent=parent)
    # if x_rep != y_rep:
    #     if x_rep[0] < y_rep[0]:
    #         parent[x_rep] = y_rep
    #     elif x_rep[0] > y_rep[0]:
    #         parent[y_rep] = x_rep
    #     else:
    #         if x_rep[1] < y_rep[1]:
    #             parent[x_rep] = y_rep
    #         elif x_rep[1] < y_rep[1]:
    #             parent[y_rep] = x_rep


def _get_seg(pixel_cls, link_cls):
    link_neighbors = pixel_cls & link_cls
    pixel_points = list(zip(*np.where(pixel_cls)))
    parent = dict.fromkeys(pixel_points, -1)

    h, w = pixel_cls.shape
    for center in pixel_points:
        neighbors = get_neighbors(center)
        # if center != (26, 31):
        #     continue
        for dir_idx, neighbor in enumerate(neighbors):
            # print(dir_idx, neighbor, link_neighbors[dir_idx, center[0], center[1]])
            if (neighbor[0] < 0) or (neighbor[0] >= h) or (neighbor[1] < 0) or (neighbor[1] >= w):
                continue
            # if (pixel_cls[neighbor[0], neighbor[1]] == 1) and (link_neighbors[dir_idx, center[0], center[1]] == 1):
            if link_neighbors[dir_idx, neighbor[0], neighbor[1]] == 1:
                _union(x=center, y=neighbor, parent=parent)
    # set(parent.values())
    # set([i[0] for i in parent.keys()]), set([i[1] for i in parent.keys()])
    # parent[(26, 31)]

    out = np.zeros(pixel_cls.shape, dtype="int32")
    root_map = dict()
    for h_idx, w_idx in pixel_points:
        root = _find((h_idx, w_idx), parent=parent)
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
    l = round(quad[:, 0].min())
    t = round(quad[:, 1].min())
    r = round(quad[:, 0].max())
    b = round(quad[:, 1].max())
    return (l, t, r, b)


def mask_to_bbox(
    pixel_pred, link_pred, mode="2s", pixel_thresh=0.5, link_thresh=0.5, area_thresh=100,
):
    """
    pixel_pred:
        Shape: (2, h, w)
        값이 작을수록 박스는 커짐
    link_pred:
        Shape: (16, h, w)
        # 값이 작을수록 박스는 커짐
    """
    # mode="2s"
    # area_thresh=100
    # pixel_thresh=0.6
    # link_thresh=0.5
    # pixel_pred=pixel_pred[0]
    # link_pred=link_pred[0]
    # pixel_pred.shape, link_pred.shape
    pixel_pred = pixel_pred.detach().cpu().numpy()
    link_pred = link_pred.detach().cpu().numpy()

    pixel_cls = pixel_pred[1, ...] > pixel_thresh
    link_cls = link_pred[8:, ...] > link_thresh

    seg = _get_seg(pixel_cls=pixel_cls, link_cls=link_cls)

    pred_bboxes = list()
    for idx in range(1, np.max(seg) + 1):
        # box_mask = (seg == idx).astype("uint8")
        box_mask = (seg == idx)
        conf = round((pixel_pred[1, ...] * box_mask).mean().round(7) * 100_000)
        if box_mask.sum() < area_thresh:
            continue
        # print(conf)

        contours, _ = cv2.findContours(
            box_mask.astype("uint8"), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE,
        )
        quad = cv2.minAreaRect(contours[0])
        quad = cv2.boxPoints(quad)
        quad = np.clip(a=quad, a_min=0, a_max=np.max(quad))
        # if _filter_short_side(quad):
        #     continue

        scale = 2 if mode == "2s" else 4
        quad *= scale
        rect = _quad_to_rect(quad)
        pred_bboxes.append((conf, *rect))
    pred_bboxes.sort(reverse=True)
    return pred_bboxes
