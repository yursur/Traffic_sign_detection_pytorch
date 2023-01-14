"""
Hyper-params for six-class detection model
and paths, which are used in the project.
"""
import torch

## DEVICE cuda OR cpu
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

## HYPER-PARAMETERS FOR TRAINING
NUM_CLASSES = 6
NUM_EPOCHS = 50
BATCH_SIZE = 15

## any detection having confidence score below this will be discarded
CONFIDENCE_THRESHOLD = 0.4
IoU_THRESHOLD = 0.4

## PATHS
IMAGES_PATH = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-frames"
GT_PATH = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-gt"
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/train/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/train/plots"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"
TEST_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/test"
SAVE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/test"

## Save model and plot during training after every n epochs
EPOCHS_SAVE_MODEL = 2
EPOCHS_SAVE_PLOTS = 2
