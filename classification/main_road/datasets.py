import os
import pandas as pd
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image

from config import CLASS_NAME, BATCH_SIZE, DATASETS_PATH

class classification_dataset(Dataset):
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
               └── test/
                       ├── class_1/
                       │        ├── img1.jpg
                       │        ├── img2.jpg
                       │        └── img3.jpg
                       │     ...
                       ├── class_6/
                       │        ├── img1.jpg
                       │        ├── img2.jpg
                       │        └── img3.jpg
                       ├── gt_test_class_1.csv
                       │     ...
                       └── gt_test_class_6.csv
    """

    def __init__(
            self,
            root_dataset: str,
            train: bool = True,
            transform=False
    ):
        if train:
            self.images_path = os.path.join(root_dataset, 'train', CLASS_NAME)
            gt = pd.read_csv(os.path.join(root_dataset, 'train', 'gt_train_'+CLASS_NAME+'.csv'))
        else:
            self.images_path = os.path.join(root_dataset, 'test', 'blue_rect')
            gt = pd.read_csv(os.path.join(root_dataset, 'test', 'gt_test_'+CLASS_NAME+'.csv'))

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

        # print(f"ИТОГО ДЛЯ №{idx}:\n    img = {img}\n    label = {label}\n____________________________________________________________")
        return img, label

    def __len__(self):
        return len(self.imgs)


# Transformations for datasets
data_transforms = {
    'train': T.Compose([
        T.Resize(size=(225,225)),
        T.ToTensor(),
        T.Normalize(mean=[0.3677, 0.3948, 0.4517], std=[0.2350, 0.2200, 0.2068])
    ]),
    'val': T.Compose([
        T.Resize(size=(225,225)),
        T.ToTensor(),
        T.Normalize(mean=[0.3677, 0.3948, 0.4517], std=[0.2350, 0.2200, 0.2068])
    ])
}

# Creating classification train and test datasets
train_dataset = classification_dataset(root_dataset=DATASETS_PATH,
                                       train=True,
                                       transform=data_transforms['train'])
test_dataset = classification_dataset(root_dataset=DATASETS_PATH,
                                      train=False,
                                      transform=data_transforms['val'])

# Creating dataloaders for classification datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE)

print(f"Number of samples in {CLASS_NAME} train dataset: {len(train_dataset)}")
print(f"Number of samples in {CLASS_NAME} test dataset: {len(test_dataset)}")

# # mean and std calculating for normalize
# def mean_std(loader):
#   images, labels = next(iter(loader))
#   # shape of images = [b,c,w,h]
#   mean, std = images.mean([0,2,3]), images.std([0,2,3])
#   return mean, std
#
# mean, std = mean_std(train_loader)
# print("mean and std: \n", mean, std)