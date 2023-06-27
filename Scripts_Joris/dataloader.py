"""
Helper files to load BPD datasets and models

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-13

"""
# <codecell> Packages
# Import packages

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# <codecell> Settings
# Script settings

UTILITIES_SCRIPT_LOCATION = "Z:/joris.vandervorst/Scripts_Joris/Utilities.py"

DATA_FOLDER = "data"  # Specify folder name
STATIC_FILE_NAME = "testBPD3.csv"  # Specify file name

PROCESSED_DATA_FOLDER = "processed_data"  # Specify folder name
PROCESSED_STATIC_FILE_NAME = "testBPD3_cleaned.pkl"  # Specify file name

MODEL_FOLDER = "models"
METRICS_FOLDER = "metrics"

RANDOM_STATE = 42


# <codecell> Import data


# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython
def load_static(file_location=None, cleaned_data=False):
    if file_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent

        if cleaned_data:
            file_location = (
                project_location / PROCESSED_DATA_FOLDER / PROCESSED_STATIC_FILE_NAME
            )
        else:
            file_location = project_location / DATA_FOLDER / STATIC_FILE_NAME

    # Change loading function based on suffix
    if file_location.suffix == ".csv":
        # Read csv file and return pandas file with data
        return pd.read_csv(file_location)

    if file_location.suffix == ".pkl":
        # Read pkl file and return pandas file with data
        with open(file_location, "rb") as f:
            return pickle.load(f)


# <codecell> Load model
def load_model(model_file):
    # Covert model file name to path
    model_file = Path(model_file)

    # Check if path is absolute,
    # otherwise use relative location defined in this script
    if not model_file.is_absolute():
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent

        # Choose most recent file of model name
        # In oder to pick specific time stamp specify exact name
        # otherwise the most recently created file that matches the regex is chosen
        model_folder_location = project_location / MODEL_FOLDER
        model_files = model_folder_location.glob(f"{model_file}*.pkl")
        model_file = max(model_files, key=lambda item: item.stat().st_ctime)

    # Load model from pickle file
    with open(model_file, "rb") as f:
        return pickle.load(f)
