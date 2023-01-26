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
DATASETS_PATH = "C:/Users/yuras/Projects/Signs/data/classification/rtsd-r3"
SAVE_MODEL_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/train/models"
SAVE_PLOTS_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/train/plots"
INFERENCE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/inference"
TEST_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/input/test"
SAVE_IMAGE_PATH = "C:/Users/yuras/Projects/Traffic_sign_detection_pytorch/classification/output/test"

## CLASSES IN SUBCLASSES
SUBCLASSES_DICT = {
    'blue_border': ['5_16', '7_3', '7_2', '7_12', '7_4', '7_11', '7_7', '5_18', '7_5', '7_6', '7_1'],
    'blue_rect': ['5_20', '5_19_1', '5_15_5', '6_3_1', '6_7', '5_15_3', '6_4', '6_6', '5_15_1',\
                  '5_15_2', '5_6', '5_5', '5_15_2_2', '5_22', '5_3', '6_2_n50', '6_2_n70',\
                  '5_15_7', '5_14', '5_21', '6_2_n60', '5_7_1', '5_7_2', '5_11', '5_8'],
    'danger': ['1_23', '1_17', '1_20_3', '1_25', '1_33', '1_15', '1_19', '1_16', '1_11_1',\
               '1_22', '1_27', '2_3_2', '1_8', '2_3', '2_3_3', '1_11', '1_12_2', '1_20',\
               '1_12', '1_2', '1_20_2', '1_21', '1_13', '1_14', '1_18', '1_1', '1_5'],
    'main_road': ['2_1', '2_2'],
    'mandatory': ['4_2_1', '4_1_6', '4_2_3', '4_1_1', '4_1_2_1', '4_1_2', '4_1_4', '4_3',\
                  '4_1_5', '4_2_2', '4_1_3', '4_1_2_2'],
    'prohibitory': ['3_24_n40', '3_24_n20', '3_4_n8', '3_4_1', '3_27', '3_18', '3_24_n5',\
                    '3_24_n30', '3_24_n60', '3_24_n70', '3_24_n50', '3_32', '2_5', '3_1',\
                    '3_20', '3_13_r4.5', '3_2', '3_24_n80', '3_10', '3_28', '3_24_n10',\
                    '2_6', '3_18_2', '3_19', '3_30', '3_29', '3_11_n5', '3_13_r3.5']
}

## Save model and plot during training after every n epochs
EPOCHS_SAVE_MODEL = 2
EPOCHS_SAVE_PLOTS = 2
