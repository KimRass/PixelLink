# "Pixels within text instances are labeled as positive (i.e., text pixels), and otherwise are labeled as negative (i.e., nontext pixels)."

# "For a given pixel and one of its eight neighbors, if they belong to the same instance,
# the link between them is positive. Otherwise negative. Note that ground truth calculation
# is carried out on input images resized to the shape of prediction layer, i.e., 'conv3_3' for '4s'
# and 'conv2_2' for '2s'."

# "Given predictions on pixels and links, two different thresholds can be applied on them separately.
# Positive pixels are then grouped together using positive links, resulting in a collection of CCs,
# each representing a detected text instance. Given two neighboring positive pixels, their link
# are predicted by both of them, and they should be connected when one or both of the two link predictions
# are positive. This linking process can be implemented using disjoint-set data structure."

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def get_bboxes(csv_path):
    bboxes = pd.read_csv(csv_path, usecols=["xmin", "ymin", "xmax", "ymax"])
    for col in bboxes.columns.tolist():
        bboxes[col] = bboxes[col].apply(lambda x: round(x / 2))
    return bboxes


def load_image(csv_path):
    bboxes = pd.read_csv(csv_path, usecols=["image_url"])
    img_path = bboxes["image_url"][0]
    image = Image.open(BytesIO(requests.get(img_path).content)).convert("RGB")
    w, h = image.size
    image = image.resize(size=(w // 2, h // 2))
    img = np.array(image)
    return img


# "Pixels inside text bounding boxes are labeled as positive. If overlapping exists,
# only un-overlapped pixels are positive. Otherwise negative."
def get_pixel_gt(w, h, bboxes): # Index 0: None-text, Index 1: Text
    canvas = np.zeros((h, w), dtype="uint8")
    for row in bboxes.itertuples():
        canvas[row.ymin: row.ymax, row.xmin: row.xmax] += 1
    pixel_gt = (canvas == 1)
    pixel_gt = torch.Tensor(pixel_gt).long()
    return pixel_gt


def get_text_seg_map(w, h, pixel_gt):
    # canvas = np.zeros((h, w), dtype="uint16")
    canvas = np.zeros((h, w), dtype="uint8")
    for idx, row in enumerate(bboxes.itertuples(), start=1):
        canvas[row.ymin: row.ymax, row.xmin: row.xmax] = idx
    return canvas * pixel_gt


def get_pos_links(seg_map, stride=5):
    ls = list()
    for shift in [
        (0, stride), # "Left"
        (-stride, stride), # "Left-down"
        (stride, stride), # "Left-up"
        (0, -stride), # "Right"
        (-stride, -stride), # "Right-down"
        (stride, -stride), # "Right-up"
        (stride, 0), # "Up"
        (-stride, 0), # "Down"
    ]:
        shifted = np.roll(seg_map, shift=shift, axis=(0, 1))
        shifted = (seg_map == shifted) * pixel_gt

        ls.append(shifted)
    stacked = np.stack(ls)
    return stacked


csv_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471.csv"
img_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471_ori.jpg"

bboxes = get_bboxes(csv_path)
img = load_image(csv_path)
h, w, _ = img.shape

pixel_gt = get_pixel_gt(w=w, h=h, bboxes=bboxes)
pixel_gt = pixel_gt[None, ...].repeat(4, 1, 1)

seg_map = get_text_seg_map(w=w, h=h, pixel_gt=pixel_gt)
pos_links = get_pos_links(seg_map=seg_map, stride=1)
temp = (sum(pos_links) == 8)
show_image(img, temp)

# show_image(seg_map, pos_links[7])
show_image(img, pos_links[7])


_, out = cv2.connectedComponents(image=temp.astype("uint8"), connectivity=4)
show_image(out)




# cls_logits = torch.stack((torch.Tensor(~pixel_gt), torch.Tensor(pixel_gt)))[None, ...].repeat(2, 1, 1, 1)
# link_logits = torch.cat(
#     (torch.stack([torch.Tensor(~i) for i in pos_links]), torch.stack([torch.Tensor(i) for i in pos_links])),
#     dim=0
# )[None, ...].repeat(2, 1, 1, 1)
pixel_pos_scores = torch.Tensor(pixel_gt)[None, ...].repeat(2, 1, 1)
link_pos_scores = torch.stack([torch.Tensor(i) for i in pos_links])[None, ...].repeat(2, 1, 1, 1)
# mask, bboxes = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
mask = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
# np.unique(mask)
show_image(mask)
show_image(img, mask)
save_image(mask, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output.png")
save_image(mask, img, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output2.png")
