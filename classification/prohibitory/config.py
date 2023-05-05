import torch

## DEVICE cuda OR cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# LIST OF CLASSES OF BLUE_BORDER SUBCLASS
CLASS_NAME = 'prohibitory'
CLASSES = ['3_24_n40', '3_24_n20', '3_4_n8', '3_4_1', '3_27', '3_18', '3_24_n5',\
           '3_24_n30', '3_24_n60', '3_24_n70', '3_24_n50', '3_32', '2_5', '3_1',\
           '3_20', '3_13_r4.5', '3_2', '3_24_n80', '3_10', '3_28', '3_24_n10',\
           '2_6', '3_18_2', '3_19', '3_30', '3_29', '3_11_n5', '3_13_r3.5']

# HYPER-PARAMETERS FOR TRAINING
NUM_EPOCHS = 25
BATCH_SIZE = 30

# PATHS
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/prohibitory/output/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/prohibitory/output/plots"
DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"

# PATHS FOR TRAINING IN GOOGLE COLAB NOTEBOOK
# SAVE_MODEL_PATH = "/content/drive/MyDrive/classification/prohibitory/output/models"
# SAVE_PLOTS_PATH = "/content/drive/MyDrive/classification/prohibitory/output/plots"
# DATASETS_PATH = "/content/drive/MyDrive/classification/rtsd-r3"
