import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/text_segmenter")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from PIL import Image
import torchvision.transforms.functional as TF

import config
from model import PixelLink2s
from data import MenuImageDataset
from loss import InstanceBalancedCELoss

print(f"""SEED = {config.SEED}""")
print(f"""N_WORKERS = {config.N_WORKERS}""")
print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")

model = PixelLink2s().to(config.DEVICE)

crit = InstanceBalancedCELoss()

optim = SGD(
    params=model.parameters(),
    lr=config.INIT_LR,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY,
)
optim.param_groups[0]["lr"] = config.FIN_LR

train_ds = MenuImageDataset(csv_dir=config.CSV_DIR, split="train")
train_dl = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    num_workers=config.N_WORKERS,
    pin_memory=True,
    drop_last=True,
)
val_ds = MenuImageDataset(csv_dir=config.CSV_DIR, split="val")
val_dl = DataLoader(
    val_ds, batch_size=1, num_workers=config.N_WORKERS, pin_memory=True, drop_last=True
)


if __name__ == "__main__":
    N_EPOCHS = 100
    for epoch in range(1, N_EPOCHS + 1):
        for step, batch in enumerate(train_dl, start=1):
            image = batch["image"].to(config.DEVICE)

            pixel_gt = batch["pixel_gt"].to(config.DEVICE)
            pixel_weight = batch["pixel_weight"].to(config.DEVICE)
            # image.shape, pixel_gt.shape, pixel_weight.shape

            optim.zero_grad()
            pixel_pred = model(image)
            loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
            loss.backward()
            optim.step()

            ## Validate.
            # for batch in enumerate(val_dl, start=1):
            val_data = val_ds[0]
            val_image = val_data["image"].to(config.DEVICE)
            val_pixel_gt = val_data["pixel_gt"].to(config.DEVICE)
            # val_image.shape, val_pixel_gt.shape

            val_pixel_pred = model(val_image.unsqueeze(0))
            val_pixel_pred = (val_pixel_pred[:, 1, ...] >= 0.5).long()
            # val_pixel_gt.shape, val_pixel_pred.shape
            iou = ((val_pixel_gt == 1) & (val_pixel_pred == 1)).sum() / ((val_pixel_gt == 1) | (val_pixel_pred == 1)).sum()
            print(f"""[ {epoch} ][ {step} ][ Loss: {loss.item():.4f} ][ IoU: {iou.item():.3f} ]""")