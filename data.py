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


csv_path = "/Users/jongbeomkim/Desktop/workspace/scene_text_renderer/textual_attribute_recognition/koen/data/701_2471.csv"
img_path = "/Users/jongbeomkim/Desktop/workspace/scene_text_renderer/textual_attribute_recognition/koen/data/701_2471_ori.jpg"

bboxes = pd.read_csv(csv_path)

h, w, _ = img.shape
summed = np.zeros((h, w, 1), dtype="uint16")
textboxes = list()
for row in bboxes.itertuples():
    canvas = np.zeros((h, w, 1), dtype="uint8")
    canvas[row.ymin: row.ymax, row.xmin: row.xmax] = 1
    textboxes.append(canvas)
    summed += canvas
temp = ((summed == 1).astype("uint8") * 255)[..., 0]
show_image(img, temp, 0.7)

image = Image.open(img_path).convert("RGB")
img = np.array(image)

canvas = np.zeros_like(img, dtype="uint8")
