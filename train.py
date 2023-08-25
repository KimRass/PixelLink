import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/text_segmenter")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler
from PIL import Image
from time import time
from pathlib import Path

import config
from model import PixelLink2s
from data import MenuImageDataset
from loss import InstanceBalancedCELoss
from evaluate import get_pixel_iou
from utils import get_elapsed_time


# def validate(model, val_dl):
#     model.eval()
#     with torch.no_grad():
#         for batch in enumerate(val_dl, start=1):
#             # val_data = val_ds[0]
#             image = batch["image"].to(config.DEVICE)
#             pixel_gt = batch["pixel_gt"].to(config.DEVICE)
#             # image.shape, pixel_gt.shape

#             pixel_pred = model(image.unsqueeze(0))
#             iou = get_pixel_iou(pixel_pred, pixel_gt)
#             print(f"""[ {epoch} ][ {step} ][ Loss: {loss.item():.4f} ][ IoU: {iou.item():.3f} ]""")
#     model.train()


def save_checkpoint(epoch, model, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
    }

    torch.save(ckpt, str(save_path))


if __name__ == "__main__":
    print(f"""SEED = {config.SEED}""")
    print(f"""N_WORKERS = {config.N_WORKERS}""")
    print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")
    print(f"""DEVICE = {config.DEVICE}""")

    model = PixelLink2s(pretrained_vgg16=config.PRETRAINED_VGG16).to(config.DEVICE)

    crit = InstanceBalancedCELoss()

    optim = SGD(
        params=model.parameters(),
        lr=config.INIT_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    optim.param_groups[0]["lr"] = config.FIN_LR

    scaler = GradScaler(enabled=True if config.AUTOCAST else False)

    train_ds = MenuImageDataset(csv_dir=config.CSV_DIR, area_thresh=config.AREA_THRESH, split="train")
    train_dl = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_ds = MenuImageDataset(csv_dir=config.CSV_DIR, area_thresh=config.AREA_THRESH, split="val")
    val_dl = DataLoader(
        val_ds, batch_size=2, num_workers=config.N_WORKERS, pin_memory=True, drop_last=True
    )

    start_time = time()
    best_iou = 0
    prev_ckpt_path = ".pth"
    for epoch in range(1, config.N_EPOCHS + 1):
        running_loss = 0
        # loss_cnt = 0
        for step, batch in enumerate(train_dl, start=1):
            image = batch["image"].to(config.DEVICE)

            pixel_gt = batch["pixel_gt"].to(config.DEVICE)
            pixel_weight = batch["pixel_weight"].to(config.DEVICE)

            with torch.autocast(
                device_type=config.DEVICE.type,
                dtype=torch.float16,
                enabled=True if config.AUTOCAST else False,
            ):
                pixel_pred, link_pred = model(image)
                loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
            optim.zero_grad()
            if config.AUTOCAST:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            running_loss += loss.item()
            # loss_cnt += 1


        ### Validate.
        if (epoch % config.N_VAL_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            model.eval()
            with torch.no_grad():
                val_data = val_ds[0]
                val_image = val_data["image"].to(config.DEVICE)
                val_pixel_gt = val_data["pixel_gt"].to(config.DEVICE)

                val_pixel_pred = model(val_image.unsqueeze(0))
                iou = get_pixel_iou(val_pixel_pred, val_pixel_gt)
                print(f"""[ {epoch} ][ {step} ][ Loss: {running_loss / len(train_dl):.4f} ]""", end="")
                print(f"""[ {get_elapsed_time(start_time)} ][ IoU: {iou:.4f} ]""")

            if iou > best_iou:
                cur_ckpt_path = config.CKPT_DIR/f"""epoch_{epoch}.pth"""
                save_checkpoint(
                    epoch=epoch, model=model, optim=optim, scaler=scaler, save_path=cur_ckpt_path,
                )
                print(f"""Saved checkpoint.""")
                prev_ckpt_path = Path(prev_ckpt_path)
                if prev_ckpt_path.exists():
                    prev_ckpt_path.unlink()

                best_iou = iou
                prev_ckpt_path = cur_ckpt_path

        model.train()
