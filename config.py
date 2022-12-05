"""Hyper-params for model"""
import torch

#DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
#HYPER-PARAMETERS FOR TRAINING
NUM_CLASSES = 6
NUM_EPOCHS = 100
BATCH_SIZE = 30

#ROOTS
ROOT_IMAGE = "detection/rtsd-d3-frames"
ROOT_GT = "detection/rtsd-d3-gt"
SAVE_MODEL_ROOT = "Traffic_sign_detection_pytorch/output/train/models"
SAVE_PLOTS_ROOT = "Traffic_sign_detection_pytorch/output/train/plots"

#save model after every n epochs
EPOCHS_SAVE_MODEL = 2
EPOCHS_SAVE_PLOTS = 4
