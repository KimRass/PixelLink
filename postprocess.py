import torch
import cv2
import numpy as np
import torch.nn as nn


def get_neighbors(h_index, w_index):
    res = list()
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res


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
    pixel_cls = pixel_cls.cpu().numpy()
    link_cls = link_cls.cpu().numpy()

    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)

    for point in pixel_points:
        h_index, w_index = point
        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if nh_index < 0 or nw_index < 0 or nh_index >= h or nw_index >= w:
                continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                _union(x=point, y=neighbor, group_mask=group_mask)

    res = np.zeros(pixel_cls.shape, dtype=np.uint8)
    root_map = dict()
    for point in pixel_points:
        h_index, w_index = point
        root = _find(point, group_mask=group_mask)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        res[h_index, w_index] = root_map[root]

    return res


def _get_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _filter_short_side(bbox):
    for i, point in enumerate(bbox):
        if _get_dist(point, bbox[(i+1)%4]) < 5**2:
            return True # ignore it
    return False # do not ignore


def mask_to_bbox(pixel_pred, link_pred, n_neighbors=8, scale=2, thresh=100):
    """
    pixel_pred: batch_size * 2 * H * W
    link_pred: batch_size * 16 * H * W
    """
    n_neighbors=8
    pixel_pred = torch.randn(2, 2, 512, 512)
    link_pred = torch.randn(2, 16, 512, 512)

    batch_size, _, mask_height, mask_width = pixel_pred.shape
    # pixel_class = nn.Softmax2d()(pixel_pred)
    pixel_class = pixel_class[:, 1] > 0.7
    link_neighbors = torch.zeros(
        [batch_size, n_neighbors, mask_height, mask_width],
        dtype=torch.uint8,
        device=pixel_pred.device,
    )
    
    # for i in range(n_neighbors):
    #     temp = nn.Softmax2d()(link_pred[:, [2 * i, 2 * i + 1]])
    #     link_neighbors[:, i] = temp[:, 1] > 0.7
    #     link_neighbors[:, i] = link_neighbors[:, i] & pixel_class

    all_bboxes = list()
    for i in range(batch_size):
        res_mask = func(pixel_class[i], link_neighbors[i])
        box_num = np.amax(res_mask)
        bboxes = list()
        for i in range(1, box_num + 1):
            box_mask = (res_mask == i).astype("uint8")
            # if box_mask.sum() < thresh:
            #     pass

            # box_mask, contours, _ = cv2.findContours(
            contours, _ = cv2.findContours(
                box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE,
            )
            bbox = cv2.minAreaRect(contours[0])
            bbox = cv2.boxPoints(bbox)
            if _filter_short_side(bbox):
                pass
                continue

            bbox = np.clip(bbox * scale, 0, 128 * scale - 1).astype("uint8")
            bboxes.append(bbox)
        all_bboxes.append(bboxes)
    all_bboxes
    return all_bboxes


if __name__ == "__main__":
    out = mask_to_bbox(pixel_pred, link_pred, n_neighbors=8, scale=2)
    out
