import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision.transforms as T
from pathlib import Path
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks


device = torch.device("cpu")


def show(image):
    TF.to_pil_image(image).show()


image_1 = read_image("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg")
image_2 = read_image("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg")
image_list = [T.Resize(size=(500,500), antialias=True)(image_1),T.Resize(size=(500,500), antialias=True)(image_2)]
grid = make_grid(image_list)
show(grid)


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

transforms = weights.transforms()
images = [transforms(d).to(device) for d in image_list]
for image in images:
    print(image.shape)

output = model(images)


def inspect_model_output(output):
    for index,prediction in enumerate(output):
        print(f'Input {index + 1} has { len(prediction.get("scores")) } detected instances')
inspect_model_output(output)


def filter_model_output(output,score_threshold):
    filtred_output = list()
    for image in output:
        filtred_image = dict()
        for key in image.keys():
            filtred_image[key] = image[key][image['scores'] >= score_threshold]
        filtred_output.append(filtred_image)
    return filtred_output


def get_boolean_mask(output):
    for index,pred in enumerate(output):
        output[index]['masks'] = pred['masks'] > 0.5
        output[index]['masks'] = output[index]['masks'].squeeze(1)
    return output


output[0].keys()
output[0]["scores"].shape
output[0]["boxes"].shape
output[0]["labels"]

score_threshold = .8
output = filter_model_output(output=output,score_threshold=score_threshold)
output = get_boolean_mask(output)

for image, prediction in zip(image_list, output):
    image.shape, prediction["masks"].shape
    drawn = draw_segmentation_masks(image=image, masks=prediction["masks"], alpha=0.9)
    show(drawn)
model