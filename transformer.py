import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

def Resize_img_bb(img, bb, size):
    """Resize image and bounding boxes"""
    if len(bb) == 0:
        return img, bb

    resized_bb = []
    PIL_to_tensor = T.PILToTensor()
    Tensor_to_PIL = T.ToPILImage()
    Resize = T.Resize(size)

    for box in bb:
        temp_b = []
        temp_b.append(box[0] * size[0] / img.size[0])
        temp_b.append(box[1] * size[1] / img.size[1])
        temp_b.append(box[2] * size[0] / img.size[0])
        temp_b.append(box[3] * size[1] / img.size[1])
        resized_bb.append(temp_b)
    resized_bb = torch.as_tensor(resized_bb)
    img = PIL_to_tensor(img)
    img = Resize(img)
    img = Tensor_to_PIL(img)

    return img, resized_bb


def crop(img, boxes):
    """Convert img to cv2-format, crop it and convert back to PIL-format"""

    cropped_imgs = []
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for box in boxes:
        box = box.to(torch.int32)
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img = Image.fromarray(cropped_img)
        cropped_imgs.append(cropped_img)

    return cropped_imgs