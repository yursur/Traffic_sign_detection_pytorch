import pandas as pd

from config_classification import DATASETS_PATH



def num_to_class(num, nums_to_classes_df):
    """
    Returns the class corresponding to the number.
    """
    return nums_to_classes_df[nums_to_classes_df.class_number == num].values[0][1]