import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL.Image import Image
from sklearn.metrics import precision_score, recall_score


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def show_img_with_bb(img, bb, labels, width=5):
    """Draw image (PIL or Tensor) with all bounding boxes"""
    PIL_to_tensor = T.PILToTensor()
    Tensor_to_PIL = T.ToPILImage()
    if isinstance(img, Image):
        img_tensor = PIL_to_tensor(img)
    else: img_tensor = img.to(torch.uint8)
    # May be no one box on the picture
    try:
        img_tensor = draw_bounding_boxes(img_tensor, bb, labels=labels ,colors='red', width=width)
    except UserWarning:
        pass
    img_pil = Tensor_to_PIL(img_tensor)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_pil)

def get_class_dict(gt_df: pd.DataFrame):
    """Returns dict, like {'2_1': 14; ...} from GT DataFrame"""
    class_dict = {}
    for cl, sign in enumerate(gt_df['sign_class'].unique()):
        class_dict.setdefault(sign, cl + 1)
    return class_dict

def give_sings_from_dict(number: int, class_dict: dict):
    """
    Returns sign by id in class_dict. (0 - background)
    """
    if number == 0:
        return 0
    for sign, sign_id in class_dict.items():
        if sign_id == number:
            return sign

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def intersection_over_union(gt_box, pred_box):
    """
    Returns IoU
    """
    # get coordinates of bottom_left and top_right points of intersection
    inter_box_bottom_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_top_right = [min(gt_box[2], pred_box[2]), min(gt_box[3], pred_box[3])]
    # calculate size of intersection
    inter_box_w = inter_box_top_right[0] - inter_box_bottom_left[0]
    inter_box_h = inter_box_top_right[1] - inter_box_bottom_left[1]
    intersection = inter_box_w * inter_box_h
    # claculate size of ground truth box
    gt_size = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    # claculate size of predicted box
    pred_size = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    # calculate size of union
    union = gt_size + pred_size - intersection

    iou = intersection / union

    return iou

def precision_recall(y_true, pred_scores, thresholds):
    """
    Returns list of precisions and list of recalls for every threshold in thresholds list
    """
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precisions.append(precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive"))
        recalls.append(recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive"))

    return precisions, recalls