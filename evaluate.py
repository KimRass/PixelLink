import torch


def get_pixel_iou(pixel_pred, pixel_gt):
    # pixel_pred = torch.randn(1, 2, 1768, 1256)
    # pixel_gt = torch.randint(high=2, size=(1768, 1256))
    argmax = torch.argmax(pixel_pred, dim=1, keepdim=True)

    pred_mask = (argmax == 1)
    gt_mask = (pixel_gt == 1)
    if gt_mask.sum().item() == 0:
        return 0

    union = (pred_mask | gt_mask).sum().item()
    intersec = (pred_mask & gt_mask).sum().item()
    iou = intersec / union
    return iou
