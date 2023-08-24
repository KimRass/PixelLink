import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def draw_bboxes(image: Image.Image, bboxes: pd.DataFrame) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for row in bboxes.itertuples(): # Draw bboxes
        draw.rectangle(
            xy=(row.x1, row.y1, row.x2, row.y2),
            outline="rgb(255, 0, 0)",
            fill=None,
            width=2,
        )
        draw.line(xy=(row.x1, row.y1, row.x2, row.y2), fill="rgb(255, 0, 0)", width=1)
        draw.line(xy=(row.x1, row.y2, row.x2, row.y1), fill="rgb(255, 0, 0)", width=1)
    return canvas


def pos_pixel_mask_to_pil(pos_pixel_mask):
    return Image.fromarray((pos_pixel_mask.numpy().astype("uint8") * 255))


if __name__ == "__main__":
    # image.min(), image.max()
    grid = make_grid(val_image, normalize=True, value_range=(-1, 1))
    TF.to_pil_image(grid).show()

    grid = make_grid(val_pixel_gt.float())
    TF.to_pil_image(grid).show()

    val_pixel_gt.shape, val_pixel_pred.shape

    val_pixel_pred.sum(dim=1)
    # grid = make_grid((val_pixel_pred[:, 1, ...] >= 0.5).float())
    grid = make_grid((val_pixel_pred[:, 0, ...] >= 0.5).float())
    # grid.min(), grid.max()
    TF.to_pil_image(grid).show()