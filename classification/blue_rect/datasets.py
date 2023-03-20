import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

from classification.datasets import ClassificationDataset
from config import CLASS_NAME, BATCH_SIZE, DATASETS_PATH

# Transformations for datasets
data_transforms = {
    'train': T.Compose([T.RandomAutocontrast(),
                        T.RandomAdjustSharpness(sharpness_factor=0),
                        T.Resize(size=(225, 225)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.3677, 0.3948, 0.4517], std=[0.2350, 0.2200, 0.2068])
                        ]),
    'val': T.Compose([T.Resize(size=(225, 225)),
                      T.ToTensor(),
                      T.Normalize(mean=[0.3677, 0.3948, 0.4517], std=[0.2350, 0.2200, 0.2068])
                      ])
}

# Creating classification train and test datasets
train_dataset = ClassificationDataset(class_name=CLASS_NAME,
                                      root_dataset=DATASETS_PATH,
                                      train=True,
                                      transform=data_transforms['train'])
test_dataset = ClassificationDataset(class_name=CLASS_NAME,
                                     root_dataset=DATASETS_PATH,
                                     train=False,
                                     transform=data_transforms['val'])

# Creating dataloaders for classification datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE)

print(f"Number of samples in {CLASS_NAME} train dataset: {len(train_dataset)}")
print(f"Number of samples in {CLASS_NAME} test dataset: {len(test_dataset)}")

# mean and std calculating for normalize
# def mean_std(loader):
#   images, labels = next(iter(loader))
#   # shape of images = [b,c,w,h]
#   mean, std = images.mean([0,2,3]), images.std([0,2,3])
#   return mean, std
#
# mean, std = mean_std(train_loader)
# print("mean and std: \n", mean, std)
