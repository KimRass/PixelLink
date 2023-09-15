import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    return neighbors


def _find(x, group_mask):
    root = x
    while group_mask.get(root) != -1:
        root = group_mask.get(root)
    return root


def _union(x, y, group_mask):
    roota = _find(x, group_mask=group_mask)
    rootb = _find(y, group_mask=group_mask)
    if roota != rootb:
        group_mask[rootb] = roota
    return


def func(pixel_cls, link_cls):
    # pixel_cls = pixel_pred_temp[i]
    # link_cls = link_neighbors[i]
    pixel_cls = pixel_cls.detach().cpu().numpy()
    link_cls = link_cls.detach().cpu().numpy()

    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)

    for point in pixel_points:
        h_idx, w_idx = point
        neighbors = get_neighbors(h_idx, w_idx)
        for i, neighbor in enumerate(neighbors):
            nh_idx, nw_idx = neighbor
            if nh_idx < 0 or nw_idx < 0 or nh_idx >= h or nw_idx >= w:
                continue
            if pixel_cls[nh_idx, nw_idx] == 1 and link_cls[i, h_idx, w_idx] == 1:
                _union(x=point, y=neighbor, group_mask=group_mask)

    out = np.zeros(pixel_cls.shape, dtype="int32")
    root_map = dict()
    for point in pixel_points:
        h_idx, w_idx = point
        root = _find(point, group_mask=group_mask)
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
        out_mask = func(pixel_pred_temp[i], link_neighbors[i])
        # _to_pil(_repaint_segmentation_map(out_mask)).show()
        n_bboxes = np.max(out_mask)

        bboxes = list()
        for i in range(1, n_bboxes + 1):
            box_mask = (out_mask == i).astype("uint8")
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
    out = mask_to_bbox(pixel_pred, link_pred, config.N_NEIGHBORS=8, scale=2)
    out
