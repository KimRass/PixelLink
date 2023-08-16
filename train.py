import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from PIL import Image
import torchvision.transforms.functional as TF

from model import PixelLink2s
from data import MenuImageDataset
from loss import InstanceBalancedCELoss
from infer import _pad_input_image

# "Optimized by SGD with $momentum = 0.9$ and $weight_decay = 5 \times 10^{-4}$.
MOMENTUM = 0.9
WEIGHT_DECAY = 5 * 1e-4
# "Learning rate is set to $10^{-3}$ for the first 100 iterations, and fixed at $10^{-2}$ for the rest."
INIT_LR = 1e-3
FIN_LR = 1e-2
N_WORKERS = 0
BATCH_SIZE = 1
IMG_SIZE = 512
FEAT_MAP_SIZE = IMG_SIZE // 2

model = PixelLink2s()

crit = InstanceBalancedCELoss()

optim = SGD(
    params=model.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)
optim.param_groups[0]["lr"] = FIN_LR

csv_dir = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/data"
train_ds = MenuImageDataset(csv_dir=csv_dir, split="train")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
val_ds = MenuImageDataset(csv_dir=csv_dir, split="val")
# val_dl = DataLoader(val_ds, batch_size=1, num_workers=N_WORKERS, pin_memory=False, drop_last=False)

N_EPOCHS = 100
for epoch in range(1, N_EPOCHS + 1):
    for step, batch in enumerate(train_dl, start=1):
        # batch = next(iter(dl))
        image = batch["image"]

        pixel_gt = batch["pixel_gt"]
        pixel_weight = batch["pixel_weight"]
        # image.shape, pixel_gt.shape, pixel_weight.shape

        pixel_pred = model(image)
        loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
        print(f"""[ {epoch} ][ {step} ][ Loss: {loss.item():.4f}]""")

        ### Validate.
        val_data = val_ds[0]
        val_image = val_data["image"]
        val_pixel_gt = val_data["pixel_gt"]
        val_image.shape, val_pixel_gt.shape

        val_pixel_pred = model(val_image.unsqueeze(0))
        val_pixel_pred = (val_pixel_pred >= 0.5).long()
        iou = (val_pixel_gt == 1 & val_pixel_gt == 1).sum() / (val_pixel_gt == 1 | val_pixel_gt == 1).sum()
        (val_pixel_gt == 1).sum()
        ((val_pixel_gt == 1) & (val_pixel_gt == 1)).sum()
        ((val_pixel_gt == 1) | (val_pixel_gt == 1)).sum()
        # (val_pixel_gt | val_pixel_gt).sum()
        iou

        loss.backward()
        optim.step()
