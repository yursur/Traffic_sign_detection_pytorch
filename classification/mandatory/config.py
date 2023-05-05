import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# LIST OF CLASSES OF BLUE_BORDER SUBCLASS
CLASS_NAME = 'mandatory'
CLASSES = ['4_2_1', '4_1_6', '4_2_3', '4_1_1', '4_1_2_1', '4_1_2', '4_1_4', '4_3',\
           '4_1_5', '4_2_2', '4_1_3', '4_1_2_2']

# HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 25
BATCH_SIZE = 30

# PATHS
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/mandatory/output/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/mandatory/output/plots"
DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"

# PATHS FOR TRAINING IN GOOGLE COLAB NOTEBOOK
# SAVE_MODEL_PATH = "/content/drive/MyDrive/classification/mandatory/output/models"
# SAVE_PLOTS_PATH = "/content/drive/MyDrive/classification/mandatory/output/plots"
# DATASETS_PATH = "/content/drive/MyDrive/classification/rtsd-r3"
