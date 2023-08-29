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
from utils import get_elapsed_time, vis_image, vis_pixel_pred, vis_link_pred

val_ds = MenuImageDataset(csv_dir=config.CSV_DIR, area_thresh=config.AREA_THRESH, split="val")
# val_dl = DataLoader(
#     val_ds, batch_size=2, num_workers=config.N_WORKERS, pin_memory=True, drop_last=True
# )

model = PixelLink2s(pretrained_vgg16=config.PRETRAINED_VGG16).to(config.DEVICE)
ckpt_path = "/Users/jongbeomkim/Downloads/epoch_290.pth"
ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
model.load_state_dict(ckpt["model"])

data = val_ds[1]
image = data["image"].to(config.DEVICE)
pixel_gt = data["pixel_gt"].to(config.DEVICE)
# vis_image(image)

pixel_pred, link_pred = model(image.unsqueeze(0))

# vis_pixel_gt(image, pixel_gt)
vis_pixel_pred(image, pixel_pred)
# vis_link_pred(image, link_pred)