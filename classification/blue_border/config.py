import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# LIST OF CLASSES OF BLUE_BORDER SUBCLASS
CLASS_NAME = 'blue_border'
CLASSES = ['5_16', '7_3', '7_2', '7_12', '7_4', '7_11', '7_7', '5_18', '7_5', '7_6', '7_1']

# HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 25
BATCH_SIZE = 30

# PATHS
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/blue_border/output/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/blue_border/output/plots"
DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"

# PATHS FOR TRAINING IN GOOGLE COLAB NOTEBOOK
# SAVE_MODEL_PATH = "/content/drive/MyDrive/classification/blue_border/output/models"
# SAVE_PLOTS_PATH = "/content/drive/MyDrive/classification/blue_border/output/plots"
# DATASETS_PATH = "/content/drive/MyDrive/classification/rtsd-r3"
