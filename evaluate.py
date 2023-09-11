

def get_pixel_iou(pixel_pred, pixel_gt):
    # pixel_pred = torch.randn(1, 2, 1768, 1256)
    # pixel_gt = torch.randn(1, 1768, 1256)
    pixel_pred = (pixel_pred[:, 1, ...] >= 0.5).long()
    pixel_pred = (pixel_pred == 1)
    pixel_gt = (pixel_gt == 1)

    intersec = (pixel_pred & pixel_gt).sum()
    union = (pixel_pred | pixel_gt).sum()
    iou = intersec / union
    return iou.item()
