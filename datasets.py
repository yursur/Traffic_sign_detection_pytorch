import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_area
from torchvision.transforms import functional as F
from PIL import Image

from transformer import Resize_img_bb
from config_detection import IMAGES_PATH, GT_PATH, BATCH_SIZE
from utils import collate_fn

class RTSD_by_groups(Dataset):
    """
        Image dataset structured as follows:
    root_image/
               ├── train/
               │       ├── img1.jpg
               │       ├── img2.jpg
               │       └── img3.jpg
               └── test/
                        ├── img1.jpg
                        ├── img2.jpg
                        └── img3.jpg

    GT's of image dataset structured as follows:
    root_gt/
        ├── class_x
        │   ├── train_gt.csv
        │   └── test_gt.csv
        └── class_y
            ├── train_gt.csv
            └── test_gt.csv
    """

    def __init__(
            self,
            root_image: str,
            root_gt: str,
            train: bool = True,
            transform=False
    ):
        if train:
            self.root_image = os.path.join(root_image, 'train')
            imgs = pd.read_csv(os.path.join(root_gt, 'train_filenames.txt'), names=['filename'])
            self.gt_filename = 'train_gt.csv'
        else:
            self.root_image = os.path.join(root_image, 'test')
            imgs = pd.read_csv(os.path.join(root_gt, 'test_filenames.txt'), names=['filename'])
            self.gt_filename = 'test_gt.csv'

        self.imgs = list(imgs.filename)
        self.root_gt = root_gt
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root_image, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # get boxes and labels
        boxes = []
        labels = []

        for n, cl in enumerate(os.listdir(self.root_gt)):
            path = os.path.join(self.root_gt, cl)

            if os.path.isdir(path):
                gt_df = pd.read_csv(os.path.join(path, self.gt_filename))
            else:
                continue

            if self.imgs[idx] in list(gt_df['filename']):
                num_objs = gt_df['filename'].value_counts()[self.imgs[idx]]
                box = []

                for _, row in gt_df[gt_df['filename'] == self.imgs[idx]].items():
                    box.append(list(row))

                for i in range(num_objs):
                    boxes.append([box[1][i], box[2][i], box[3][i] + box[1][i], box[4][i] + box[2][i]])

                labels += [n+1] * num_objs

        # print(f'в итоге после прогона по {n+1} классу: \n boxes = {boxes}\nlabels = {labels}')

        if len(labels) == 0:
            area = torch.zeros(1)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels.append(0)
            iscrowd = torch.ones(1).to(torch.int64)
        elif len(labels) == 1:
            area = torch.as_tensor([(boxes[0][3] - boxes[0][1]) * (boxes[0][2] - boxes[0][0])], dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = box_area(boxes)
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        if self.transform:
            img, boxes = Resize_img_bb(img, boxes, size=(600, 600))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, dtype=torch.float32)
        #print(f"ИТОГО ДЛЯ №{idx}:\n    img = {img}\n    target = {target}\n____________________________________________________________")

        return img, target

    def __len__(self):
        return len(self.imgs)

train_dataset = RTSD_by_groups(root_image=IMAGES_PATH, root_gt=GT_PATH, train=True, transform=True)
test_dataset = RTSD_by_groups(root_image=IMAGES_PATH, root_gt=GT_PATH, train=False, transform=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          collate_fn=collate_fn)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
