import torch
import numpy as np
import cv2
from models import create_model
from config import DEVICE, NUM_CLASSES, SAVE_MODEL_ROOT

# load the model and the trained weights
model = create_model(num_classes=NUM_CLASSES, six_class_detection=True).to(DEVICE)
model.load_state_dict(torch.load(SAVE_MODEL_ROOT+'/detection_model14.pth', map_location=DEVICE))
model.eval()