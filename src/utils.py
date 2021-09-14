""" Supporting functions"""
import glob
import os

import pandas as pd


def read_data(filename, features):
    """ Read data from file and return dataframe for selected features
    Args:
        filename (str): name of csv file
        features (list): list of string with feature names
    Returns:
        DataFrame: data of csv file in dataframe
    """
    # Read combined dataset, note this is the same as used for training
    data = pd.read_csv(filename, encoding="iso-8859-1")
    # Drop rows with missing values for features
    df_nonan = data.dropna(subset=features, inplace=False)

    return df_nonan


def clean_dirs(outdirs):
    """ Make sure output dirs exist and are empty
    Args:
        outdirs (list): list of directory names
    """
    for dir_ in outdirs:
        if os.path.exists(dir_):
            files = glob.glob(f"{dir}/*")
            for file_ in files:
                os.remove(file_)
        else:
            os.mkdir(dir_)
