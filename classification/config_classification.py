"""
Hyper-params for classification models
and paths, which are used in the project.
"""
import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

## HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 50
BATCH_SIZE = 15

## PATHS
IMAGES_PATH = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-frames"
GT_PATH = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-gt"
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/train/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/train/plots"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"
TEST_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/test"
SAVE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/test"

## Save model and plot during training after every n epochs
EPOCHS_SAVE_MODEL = 2
EPOCHS_SAVE_PLOTS = 2
