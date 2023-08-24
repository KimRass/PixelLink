

def get_pixel_iou(pixel_pred, pixel_gt):
    pixel_pred = (pixel_pred[:, 1, ...] >= 0.5).long()
    pixel_pred = (pixel_pred == 1)
    pixel_gt = (pixel_gt == 1)

    intersec = pixel_pred & pixel_gt
    union = pixel_pred | pixel_gt
    iou = intersec / union
    return iou
