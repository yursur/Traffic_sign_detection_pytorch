import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# LIST OF CLASSES OF BLUE_BORDER SUBCLASS
CLASS_NAME = 'danger'
CLASSES = ['1_23', '1_17', '1_20_3', '1_25', '1_33', '1_15', '1_19', '1_16', '1_11_1',\
           '1_22', '1_27', '2_3_2', '1_8', '2_3', '2_3_3', '1_11', '1_12_2', '1_20',\
           '1_12', '1_2', '1_20_2', '1_21', '1_13', '1_14', '1_18', '1_1', '1_5']

# HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 25
BATCH_SIZE = 30

# PATHS
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/danger/output/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/danger/output/plots"
DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"

# PATHS FOR TRAINING IN GOOGLE COLAB NOTEBOOK
# SAVE_MODEL_PATH = "/content/drive/MyDrive/classification/danger/output/models"
# SAVE_PLOTS_PATH = "/content/drive/MyDrive/classification/danger/output/plots"
# DATASETS_PATH = "/content/drive/MyDrive/classification/rtsd-r3"

