import torch


def get_pixel_iou(pixel_pred, pixel_gt):
    pixel_pred = torch.randn(1, 2, 1768, 1256)
    pixel_gt = torch.randint(high=2, size=(1768, 1256))
    argmax = torch.argmax(pixel_pred, dim=1, keepdim=True)
    ious = list()
    for c in range(2):
        pred_mask = (argmax == c)
        gt_mask = (pixel_gt == c)
        if gt_mask.sum().item() == 0:
            continue

        union = (pred_mask | gt_mask).sum().item()
        intersec = (pred_mask & gt_mask).sum().item()
        iou = intersec / union
        ious.append(iou)
    avg_iou = sum(ious) / len(ious)
    return avg_iou
