import torch
import torchvision.transforms as T
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
