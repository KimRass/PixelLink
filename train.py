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
from postprocess import mask_to_bbox


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=False, default=0)
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def validate(model, val_dl):
    model.eval()
    accum_iou = 0
    with torch.no_grad():
        for batch in val_dl:
            image = batch["image"].to(config.DEVICE)
            pixel_gt = batch["pixel_gt"].to(config.DEVICE)

            pixel_pred, _ = model(image)
            iou = get_pixel_iou(pixel_pred, pixel_gt)

            accum_iou += iou
    avg_iou = accum_iou / len(val_dl)
    model.train()
    return avg_iou


def save_checkpoint(epoch, model, optim, scaler, best_avg_iou, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "best_average_iou": best_avg_iou,
    }
    torch.save(ckpt, str(ckpt_path))


def resume(ckpt_path, model, optim, scaler):
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        init_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        best_avg_iou = ckpt["best_average_iou"]

        prev_ckpt_path = ckpt_path

        print(f"Resume from checkpoint '{Path(ckpt_path).name}'.")
        print(f"Previous best average pixel IoU: {best_avg_iou:.3f}")
    else:
        init_epoch = 0
        prev_ckpt_path = ".pth"
        best_avg_iou = 0
    return init_epoch, prev_ckpt_path, best_avg_iou


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"""SEED = {config.SEED}""")
    # print(f"""N_WORKERS = {args.n_workers}""")
    # print(f"""BATCH_SIZE = {args.batch_size}""")
    print(f"""DEVICE = {config.DEVICE}""")
    print(f"""AMP = {config.AMP}""")

    model = PixelLink2s(pretrained_vgg16=config.PRETRAINED_VGG16).to(config.DEVICE)

    crit = InstanceBalancedCELoss()

    optim = SGD(
        params=model.parameters(),
        lr=config.INIT_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    # optim.param_groups[0]["lr"] = config.FIN_LR

    scaler = GradScaler(enabled=True if config.AMP else False)

    ds = MenuImageDataset(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        size_thresh=config.SIZE_THRESH,
        area_thresh=config.AREA_THRESH,
        split="train",
    )
    train_ds_size = round(len(ds) * 0.9)
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, lengths=(train_ds_size, val_ds_size))
    train_dl = DataLoader(
        # train_ds,
        ds,
        batch_size=args.batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    ### Resume
    init_epoch, prev_ckpt_path, best_avg_iou = resume(
        ckpt_path=args.ckpt_path, model=model, optim=optim, scaler=scaler,
    )

    # Training
    start_time = time()
    for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
        accum_pixel_loss = 0
        accum_link_loss = 0
        # for step, batch in enumerate(tqdm(train_dl, total=len(train_dl)), start=1):
        for step, batch in enumerate(train_dl, start=1):
            image = batch["image"].to(config.DEVICE)
            pixel_gt = batch["pixel_gt"].to(config.DEVICE)
            pixel_weight = batch["pixel_weight"].to(config.DEVICE)
            link_gt = batch["link_gt"].to(config.DEVICE)

        #     with torch.autocast(
        #         device_type=config.DEVICE.type,
        #         dtype=torch.float16 if config.DEVICE.type == "cuda" else torch.bfloat16,
        #         enabled=True if config.AMP else False,
        #     ):
        #         pixel_pred, link_pred = model(image)
        #         # out = mask_to_bbox(pixel_pred=pixel_pred, link_pred=link_pred)
        #         # print(out)
        #         pixel_loss, link_loss = crit(
        #             pixel_pred=pixel_pred,
        #             pixel_gt=pixel_gt,
        #             pixel_weight=pixel_weight,
        #             link_pred=link_pred,
        #             link_gt=link_gt,
        #         )
        #         loss = config.LAMB * pixel_loss + link_loss
        #     optim.zero_grad()
        #     if config.AMP:
        #         scaler.scale(loss).backward()
        #         scaler.step(optim)
        #         scaler.update()
        #     else:
        #         loss.backward()
        #         optim.step()

        #     accum_pixel_loss += pixel_loss.item()
        #     accum_link_loss += link_loss.item()

        # ### Validate.
        # avg_iou = validate(model=model, val_dl=val_dl)

        # print(f"""[ {epoch} ][ {step} ][ {get_elapsed_time(start_time)} ]""", end="")
        # print(f"""[ Pixel loss: {accum_pixel_loss / len(train_dl):.4f} ]""", end="")
        # print(f"""[ Link loss: {accum_link_loss / len(train_dl):.4f} ]""", end="")
        # print(f"""[ Average pixel IoU: {avg_iou:.3f} ]""")

        # if avg_iou > best_avg_iou:
        #     best_avg_iou = avg_iou
        #     ckpt_path = config.CKPT_DIR/f"""epoch_{epoch}.pth"""
        #     save_checkpoint(
        #         epoch=epoch,
        #         model=model,
        #         optim=optim,
        #         scaler=scaler,
        #         best_avg_iou=best_avg_iou,
        #         ckpt_path=ckpt_path,
        #     )
        #     print(f"""Saved checkpoint.""")

        #     prev_ckpt_path = Path(prev_ckpt_path)
        #     if prev_ckpt_path.exists():
        #         prev_ckpt_path.unlink()
        #     prev_ckpt_path = ckpt_path

        # start_time = time()
