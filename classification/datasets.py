import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ClassificationDataset(Dataset):
    """
    Classification dataset class.
        Image dataset structured as follows:
    root_dataset/
               ├── train/
               │       ├── class_1/
               │       │        ├── img1.jpg
               │       │        ├── img2.jpg
               │       │        └── img3.jpg
               │       │     ...
               │       ├── class_6/
               │       │        ├── img1.jpg
               │       │        ├── img2.jpg
               │       │        └── img3.jpg
               │       ├── gt_train_class_1.csv
               │       │     ...
               │       └── gt_train_class_6.csv
               ├── test/
               │       ├── class_1/
               │       │        ├── img1.jpg
               │       │        ├── img2.jpg
               │       │        └── img3.jpg
               │       │     ...
               │       ├── class_6/
               │       │        ├── img1.jpg
               │       │        ├── img2.jpg
               │       │        └── img3.jpg
               │       ├── gt_test_class_1.csv
               │       │     ...
               │       └── gt_test_class_6.csv
               └── num_to_label.csv
    """

    def __init__(
            self,
            class_name: str,
            root_dataset: str,
            train: bool = True,
            transform=False
    ):
        if train:
            self.images_path = os.path.join(root_dataset, 'train', class_name)
            gt = pd.read_csv(os.path.join(root_dataset, 'train', 'gt_train_' + class_name + '.csv'))
        else:
            self.images_path = os.path.join(root_dataset, 'test', class_name)
            gt = pd.read_csv(os.path.join(root_dataset, 'test', 'gt_test_' + class_name + '.csv'))

        self.num_to_cl_df = pd.read_csv(os.path.join(root_dataset, class_name + '_num_to_labels.csv'))

        # extract images and labels of current subclass
        self.imgs = os.listdir(self.images_path)
        self.labels = gt.label.to_list()
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        # load image and get class label
        img_path = os.path.join(self.images_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        # print(f"ИТОГО ДЛЯ №{idx}:\n    img = {img}\n    label = {label})
        # print('_'*50)
        return img, label

    def __len__(self):
        return len(self.imgs)
