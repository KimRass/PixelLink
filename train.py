import gc
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from time import time
from pathlib import Path
import argparse
from tqdm.auto import tqdm

import config
from model import PixelLink2s
from data import MenuImageDataset
from loss import InstanceBalancedCELoss
from evaluate import get_pixel_iou
from utils import get_elapsed_time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    return args


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
    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"""SEED = {config.SEED}""")
    print(f"""N_WORKERS = {config.N_WORKERS}""")
    print(f"""BATCH_SIZE = {args.batch_size}""")
    print(f"""DEVICE = {config.DEVICE}""")

    model = PixelLink2s(pretrained_vgg16=config.PRETRAINED_VGG16).to(config.DEVICE)

    crit = InstanceBalancedCELoss(lamb=config.LAMB)

    optim = SGD(
        params=model.parameters(),
        lr=config.INIT_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    optim.param_groups[0]["lr"] = config.FIN_LR

    scaler = GradScaler(enabled=True if config.AUTOCAST else False)

    ds = MenuImageDataset(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        area_thresh=config.AREA_THRESH,
        split="train",
    )
    train_ds_size = round(len(ds) * 0.9)
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, lengths=(train_ds_size, val_ds_size))
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    best_iou = 0
    prev_ckpt_path = ".pth"
    start_time = time()
    cnt = 0
    for epoch in tqdm(range(1, config.N_EPOCHS + 1)):
        running_loss = 0
        for step, batch in tqdm(enumerate(train_dl, start=1), total=len(train_dl)):
            image = batch["image"].to(config.DEVICE)

            pixel_gt = batch["pixel_gt"].to(config.DEVICE)
            pixel_weight = batch["pixel_weight"].to(config.DEVICE)
            link_gt = batch["link_gt"].to(config.DEVICE)

            with torch.autocast(
                device_type=config.DEVICE.type,
                dtype=torch.float16,
                enabled=True if config.AUTOCAST else False,
            ):
                pixel_pred, link_pred = model(image)
                loss = crit(
                    pixel_pred=pixel_pred,
                    pixel_gt=pixel_gt,
                    pixel_weight=pixel_weight,
                    link_pred=link_pred,
                    link_gt=link_gt,
                )
            optim.zero_grad()
            if config.AUTOCAST:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            running_loss += loss.item()
            cnt += 1

            ### Validate.
            if step % config.N_PRINT_STEPS == 0:
                print(f"""[ {epoch} ][ {step} ][ {get_elapsed_time(start_time)} ]""", end="")
                print(f"""[ Loss: {running_loss / cnt:.6f} ]""")

                start_time = time()
                cnt = 0
                # model.eval()
                # with torch.no_grad():
                #     val_data = val_ds[0]
                #     val_image = val_data["image"].to(config.DEVICE)
                #     val_pixel_gt = val_data["pixel_gt"].to(config.DEVICE)

                #     val_pixel_pred, val_link_pred = model(val_image.unsqueeze(0))
                #     iou = get_pixel_iou(val_pixel_pred, val_pixel_gt)
                #     print(f"""[ {epoch} ][ {step} ][ {get_elapsed_time(start_time)} ]""", end="")
                #     print(f"""[ Loss: {running_loss / len(train_dl):.4f} ][ IoU: {iou:.4f} ]""")

                #     start_time = time()

                # if iou > best_iou:
                #     cur_ckpt_path = config.CKPT_DIR/f"""epoch_{epoch}.pth"""
                #     save_checkpoint(
                #         epoch=epoch, model=model, optim=optim, scaler=scaler, save_path=cur_ckpt_path,
                #     )
                #     print(f"""Saved checkpoint.""")
                #     prev_ckpt_path = Path(prev_ckpt_path)
                #     if prev_ckpt_path.exists():
                #         prev_ckpt_path.unlink()

                #     best_iou = iou
                #     prev_ckpt_path = cur_ckpt_path

                # model.train()
