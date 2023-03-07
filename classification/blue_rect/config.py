import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# LIST OF CLASSES OF BLUE_BORDER SUBCLASS
CLASS_NAME = 'blue_rect'
CLASSES = ['5_20', '5_19_1', '5_15_5', '6_3_1', '6_7', '5_15_3', '6_4', '6_6', '5_15_1',\
           '5_15_2', '5_6', '5_5', '5_15_2_2', '5_22', '5_3', '6_2_n50', '6_2_n70',\
           '5_15_7', '5_14', '5_21', '6_2_n60', '5_7_1', '5_7_2', '5_11', '5_8']

# HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 25
BATCH_SIZE = 30

# PATHS
# SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/blue_rect/output/models"
# SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/blue_rect/output/plots"
# DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
# INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"

# PATHS FOR TRAINING IN GOOGLE COLAB NOTEBOOK
SAVE_MODEL_PATH = "/content/drive/MyDrive/classification/blue_rect/output/models"
SAVE_PLOTS_PATH = "/content/drive/MyDrive/classification/blue_rect/output/plots"
DATASETS_PATH = "/content/drive/MyDrive/classification/rtsd-r3"
