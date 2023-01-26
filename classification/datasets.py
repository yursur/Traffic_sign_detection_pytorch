import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_area
from torchvision.transforms import functional as F
from PIL import Image

from transformer import Resize_img_bb
from config_classification import DATASETS_PATH, SUBCLASSES_DICT, BATCH_SIZE
from utils import num_to_class

class blue_border_dataset(Dataset):
    """
    Classification dataset for blue_border subclass.
        Image dataset structured as follows:
    root_dataset/
               ├── train/
               │       ├── img1.jpg
               │       ├── img2.jpg
               │       └── img3.jpg
               ├── test/
               │       ├── img1.jpg
               │       ├── img2.jpg
               │       └── img3.jpg
               ├── train_gt.csv
               └── test_gt.csv
    """

    def __init__(
            self,
            root_dataset: str,
            subclass: str,
            train: bool = True,
            transform=False
    ):
        if train:
            self.images_path = os.path.join(root_dataset, 'train')
            gt = pd.read_csv(os.path.join(root_dataset, 'gt_train.csv'))
        else:
            self.images_path = os.path.join(root_dataset, 'test')
            gt = pd.read_csv(root_dataset+'/gt_test.csv')

        # extract only images of current subclass
        imgs = []
        labels = []
        for img in gt.filename:
            class_number = gt[gt.filename == img].class_number.values[0]
            if num_to_class(class_number) in SUBCLASSES_DICT[subclass]:
                imgs.append(img)
                labels.append(class_number)

        self.imgs = imgs
        self.labels = labels
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        # load image and get class label
        img_path = os.path.join(self.images_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            pass

        #print(f"ИТОГО ДЛЯ №{idx}:\n    img = {img}\n    label = {label}\n____________________________________________________________")
        return img, label

    def __len__(self):
        return len(self.imgs)

nums_to_classes_df = pd.read_csv(DATASETS_PATH+'/numbers_to_classes.csv')
train_imgs = pd.read_csv(os.path.join(DATASETS_PATH, 'train'))








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
