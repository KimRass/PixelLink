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
def get_pos_pixels(w, h, bboxes):
    canvas = np.zeros((h, w), dtype="uint8")
    for row in bboxes.itertuples():
        canvas[row.ymin: row.ymax, row.xmin: row.xmax] += 1
    return (canvas == 1)


def get_text_seg_map(w, h, pos_pixels):
    # canvas = np.zeros((h, w), dtype="uint16")
    canvas = np.zeros((h, w), dtype="uint8")
    for idx, row in enumerate(bboxes.itertuples(), start=1):
        canvas[row.ymin: row.ymax, row.xmin: row.xmax] = idx
    return canvas * pos_pixels


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
        shifted = (seg_map == shifted) * pos_pixels

        ls.append(shifted)
    stacked = np.stack(ls)
    return stacked


csv_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471.csv"
img_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/701_2471_ori.jpg"

bboxes = get_bboxes(csv_path)
img = load_image(csv_path)
h, w, _ = img.shape

pos_pixels = get_pos_pixels(w=w, h=h, bboxes=bboxes)
seg_map = get_text_seg_map(w=w, h=h, pos_pixels=pos_pixels)
pos_links = get_pos_links(seg_map=seg_map, stride=1)
temp = (sum(pos_links) == 8)
show_image(img, temp)

# show_image(seg_map, pos_links[7])
show_image(img, pos_links[7])


_, out = cv2.connectedComponents(image=temp.astype("uint8"), connectivity=4)
show_image(out)




# cls_logits = torch.stack((torch.Tensor(~pos_pixels), torch.Tensor(pos_pixels)))[None, ...].repeat(2, 1, 1, 1)
# link_logits = torch.cat(
#     (torch.stack([torch.Tensor(~i) for i in pos_links]), torch.stack([torch.Tensor(i) for i in pos_links])),
#     dim=0
# )[None, ...].repeat(2, 1, 1, 1)
pixel_pos_scores = torch.Tensor(pos_pixels)[None, ...].repeat(2, 1, 1)
link_pos_scores = torch.stack([torch.Tensor(i) for i in pos_links])[None, ...].repeat(2, 1, 1, 1)
# mask, bboxes = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
mask = to_bboxes(img, pixel_pos_scores.cpu().numpy(), link_pos_scores.cpu().numpy())
# np.unique(mask)
show_image(mask)
show_image(img, mask)
save_image(mask, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output.png")
save_image(mask, img, path="/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_output2.png")


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [
        (x - 1, y), # "Left"
        (x - 1, y + 1), # "Left-down"
        (x - 1, y - 1), # "Left-up"
        (x + 1, y), # "Right"
        (x + 1, y + 1), # "Right-down"
        (x + 1, y - 1), # "Right-up"
        (x, y - 1), # "Up"
        (x, y + 1), # "Down"
    ]


def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h


def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)

    def find_parent(point):
        return group_mask[point]
        
    def set_parent(point, parent):
        group_mask[point] = parent
        
    def is_root(point):
        return find_parent(point) == -1
    
    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        
        # For acceleration of find_root
        if update_parent:
            set_parent(point, root)
            
        return root
        
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        
        if root1 != root2:
            set_parent(root1, root2)
        
    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        
        mask = np.zeros_like(pixel_mask, dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask
    
    # join by link
    for point in points:
        y, x = point
        neighbours = get_neighbours_8(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
                link_value = link_mask[y, x, n_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))
    
    mask = get_all()
    return mask


def decode_image(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    mask =  decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold)
    return mask


def decode_batch(pixel_cls_scores, pixel_link_scores, pixel_conf_threshold=0.6, link_conf_threshold=0.9):
    batch_size = pixel_cls_scores.shape[0]
    batch_mask = list()
    for image_idx in tqdm(range(batch_size)):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
        # print(image_pos_pixel_scores.shape, image_pos_link_scores.shape)
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores, pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def get_shape(img):
    """
    return the height and width of an image
    """
    return np.shape(img)[0: 2]


def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype="float32")
    return np.cast[dtype](obj)


def resize(img, f = None, fx = None, fy = None, size = None, interpolation = cv2.INTER_LINEAR):
    """
    size: (w, h)
    """
    h, w = get_shape(img)
    if fx != None and fy != None:
        return cv2.resize(img, None, fx = fx, fy = fy, interpolation = interpolation)
        
    if size != None:
        size = cast(size, dtype="int")
#         size = (size[1], size[0])
        size = tuple(size)
        return cv2.resize(img, size, interpolation = interpolation)
    
    return cv2.resize(img, None, fx = f, fy = f, interpolation = interpolation)


def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)


def min_area_rect(xs, ys):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
        
    num_rects = xs.shape[0]
    box = np.empty((num_rects, 5))#cx, cy, w, h, theta
    for idx in range(num_rects):
        points = list(zip(xs[idx, :], ys[idx, :]))
        cnt = points_to_contour(points)
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        box[idx, :] = [cx, cy, w, h, theta]
    
    box = np.asarray(box, dtype = xs.dtype)
    return box


def find_contours(mask):
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_SIMPLE)
    return contours


def mask_to_bboxes(mask, image_shape =None, min_area = None, min_height = None, min_aspect_ratio = None):
    feed_shape = (1500, 2000)
    
    if image_shape is None:
        image_shape = feed_shape

    image_shape=image_shape[:2]
    image_h, image_w = image_shape[:]
    
    if min_area is None:
        min_area = 100
        
    if min_height is None:
        min_height = 10
    bboxes = list()
    max_bbox_idx = mask.max()
    mask = resize(img=mask, size=(image_w, image_h), interpolation=cv2.INTER_NEAREST)
    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = (mask == bbox_idx)
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)
        
        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue
        
        if rect_area < min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
    return bboxes


def to_bboxes(image_data, pixel_pos_scores, link_pos_scores):
    link_pos_scores = np.transpose(link_pos_scores,(0,2,3,1))    
    mask = decode_batch(pixel_pos_scores, link_pos_scores, 0.6, 0.9)[0, ...]
    return mask
    # bboxes = mask_to_bboxes(mask, image_data.shape)
    # return mask, bboxes