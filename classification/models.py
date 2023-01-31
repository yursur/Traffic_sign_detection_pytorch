from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from config import SUBCLASSES_DICT

def create_model(subclass: str):
    """
    Returns pre-trained ResNet classification model for subclass
    """
    # load pre-trained ResNet50 model
    resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # get the number of input features
    in_features = resnet_model.fc.in_features
    # and get the number of output features for current subclass
    out_features = len(SUBCLASSES_DICT[subclass])
    # define a new fully-connected layer for the classificator with required number of classes
    resnet_model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, out_features)
    )
    return resnet_model
