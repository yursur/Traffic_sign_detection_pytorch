"""Hyper-params for models"""
import torch

#DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
#HYPER-PARAMETERS FOR TRAINING
NUM_CLASSES = 6
NUM_EPOCHS = 50
BATCH_SIZE = 15

# any detection having confidence score below this will be discarded
DETECTION_THRESHOLD = 0.4

#ROOTS
ROOT_IMAGE = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-frames"
ROOT_GT = "C:/Users/yuras/Projects/Signs/data/detection/rtsd-d3-gt"
SAVE_MODEL_ROOT = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/train/models"
SAVE_PLOTS_ROOT = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/train/plots"
INFERENCE_IMAGE_ROOT = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"
TEST_IMAGE_ROOT = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/test"
SAVE_IMAGE_TEST = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/output/test"

#save models after every n epochs
EPOCHS_SAVE_MODEL = 2
EPOCHS_SAVE_PLOTS = 2
