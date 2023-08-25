import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import config
from model import PixelLink2s
from data import MenuImageDataset


if __name__ == "__main__":
    model = PixelLink2s().to(config.DEVICE)
    state_dict = torch.load("/Users/jongbeomkim/Downloads/pixellink_checkpoints/epoch_290.pth", map_location=config.DEVICE)
    model.load_state_dict(state_dict["model"])

    val_ds = MenuImageDataset(csv_dir=config.CSV_DIR, area_thresh=config.AREA_THRESH, split="val")
    batch = val_ds[2]
    image = batch["image"]
    pixel_gt = batch["pixel_gt"]

    pixel_pred = model(image.unsqueeze(0))
    # pixel_pred = pixel_pred.detach().cpu().numpy()
    # pixel_pred.shape
    # pixel
    # temp = split(pixel_pred)
    # temp = F.interpolate(pixel_pred, scale_factor=2, mode="nearest")
    # temp = (temp[:, 1, ...] >= 0.5).float()
    # # image.shape, pixel_pred.shape
    # TF.to_pil_image(temp).show()

    # TF.to_pil_image((image * 0.5) + 0.5).show()
