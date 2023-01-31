import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from classification.config import DATASETS_PATH, BATCH_SIZE
from classification.blue_border.config import CLASSES
from classification.utils import num_to_class

class classification_dataset(Dataset):
    """
    Classification dataset class.
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
            if num_to_class(class_number, nums_to_classes_df) in CLASSES:
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

        print(f"ИТОГО ДЛЯ №{idx}:\n    img = {img}\n    label = {label}\n____________________________________________________________")
        return img, label

    def __len__(self):
        return len(self.imgs)


# Get numbers_to_classes Dataframe for using it in num_to_class function.
# This datafraim contains links between each of the class numbers and its real class.
nums_to_classes_df = pd.read_csv(DATASETS_PATH+'/numbers_to_classes.csv')

# Creating classification train and test datasets
train_dataset = classification_dataset(root_dataset=DATASETS_PATH,
                                                      train=True)
test_dataset = classification_dataset(root_dataset=DATASETS_PATH,
                                      train=False)

# Creating dataloaders for classification datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE)

print(f"Number of samples in BLUE_BORDER train dataset: {len(train_dataset)}")
print(f"Number of samples in BLUE_BORDER test dataset: {len(test_dataset)}")

