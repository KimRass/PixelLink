import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD

from model import PixelLink2s
from data import MenuImageDataset
from loss import InstanceBalancedCELoss

# "Optimized by SGD with $momentum = 0.9$ and $weight_decay = 5 \times 10^{-4}$.
MOMENTUM = 0.9
WEIGHT_DECAY = 5 * 1e-4
INIT_LR = 1e-3
# "Learning rate is set to $10^{-3}$ for the first 100 iterations, and fixed at $10^{-2}$ for the rest."
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
ds = MenuImageDataset(csv_dir=csv_dir)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=True, drop_last=True)

N_EPOCHS = 100
for epoch in range(1, N_EPOCHS + 1):
    for step, batch in enumerate(dl, start=1):
        # batch = next(iter(dl))
        image = batch["image"]

        pixel_gt = batch["pixel_gt"]
        pixel_weight = batch["pixel_weight"]

        pixel_gt = F.interpolate(
            pixel_gt.float(), size=(FEAT_MAP_SIZE, FEAT_MAP_SIZE), mode="nearest"
        ).long()
        pixel_weight = F.interpolate(
            pixel_weight, size=(FEAT_MAP_SIZE, FEAT_MAP_SIZE), mode="nearest"
        )

        pixel_pred = model(image)
        # pixel_pred
        # image.min(), image.max()
        loss = crit(pixel_pred=pixel_pred, pixel_gt=pixel_gt, pixel_weight=pixel_weight)
        print(f"""[ {epoch} ][ {step} ][ Loss: {loss.item():.4f}]""")

        loss.backward()
        optim.step()
