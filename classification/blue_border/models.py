from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


def create_model(num_classes: int):
    """
    Returns pre-trained ResNet classification model for subclass.
    Params:
        num_classes - number of classes in the current subclass.
    """
    # load pre-trained ResNet50 model
    resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # get the number of input features
    in_features = resnet_model.fc.in_features
    # define a new fully-connected layer for the classifier with required number of classes
    resnet_model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )
    return resnet_model
