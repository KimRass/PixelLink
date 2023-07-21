from PIL import Image
import cv2
import numpy as np
import pandas as pd


seg_map = np.array(Image.open("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/SegmentationObject/2007_000129.png"))


def get_label(csv_path):
    bboxes = pd.read_csv(csv_path)

    text_img = load_image(bboxes["image_url"][0])
    w, h = _get_width_and_height(text_img)

    bboxes.sort_values(by="item_org_id", inplace=True)
    bboxes.rename(
        {"item_org_id": "bbox_id", "xmin": "bbox_x1", "ymin": "bbox_y1", "xmax": "bbox_x2", "ymax": "bbox_y2"},
        axis=1,
        inplace=True
    )
    bboxes.drop([" item_id", "image_url", "inpainting_image_url"], axis=1, inplace=True)
    bboxes.set_index("bbox_id", inplace=True)

    canvas = np.zeros((h, w)).astype("uint8")
    for idx, row in enumerate(bboxes.itertuples(), start=1):
        canvas[row.bbox_y1: row.bbox_y2, row.bbox_x1: row.bbox_x2] = idx
    return canvas
label = get_label(csv_path)
