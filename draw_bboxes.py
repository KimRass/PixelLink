from data import _get_path_pairs
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

from utils import draw_gt_bboxes
from data import get_bboxes


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def draw_bboxes(image, bboxes):
    pil_image = image.copy()
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        if len(bbox) == 5:
            bbox = bbox[1:]
        draw.rectangle(xy=bbox, outline="red", width=1)
    return pil_image


if __name__ == "__main__":
    args = get_args()

    path_pairs = _get_path_pairs(args.data_dir)
    for gt_path, img_path in tqdm(path_pairs):
        image = Image.open(img_path).convert("RGB")
        gt_bboxes = get_bboxes(gt_path)
        drawn = draw_gt_bboxes(bboxes=gt_bboxes, image=image, alpha=0.6)
        # drawn.show()
        drawn.save(Path(args.save_dir)/f"{Path(img_path).stem}_bboxes.jpg")


import torch
import torch.nn as nn
m = nn.Sigmoid()
loss = nn.BCELoss(reduction="none")
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
m(input), target

output = loss(m(input), target)
output
# nn.BCEWithLogitsLoss()(input, target)
