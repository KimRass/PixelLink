import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _pad_input_image(image):
    """
    Resize the image so that the width and the height are multiples of 16 each.
    """
    # _, _, h, w = image.shape
    w, h = image.size
    if h % 16 != 0:
        new_h = h + (16 - h % 16)
    else:
        new_h = h
    if w % 16 != 0:
        new_w = w + (16 - w % 16)
    else:
        new_w = w
    new_image = TF.pad(image, padding=(0, 0, new_w - w, new_h - h), padding_mode="constant")
    return new_image
