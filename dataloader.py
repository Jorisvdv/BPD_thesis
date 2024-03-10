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
SSPS_FILE_NAME = "static_hashed_SPSS.csv"

PROCESSED_DATA_FOLDER = "processed_data"  # Specify folder name
PROCESSED_FILE_NAME_STATIC = "testBPD3_cleaned.pkl"  # Specify file name
PROCESSED_FILE_NAME_TEMPORAL = "testBPD3_temporal.pkl"  # Specify file name
PROCESSED_FILE_NAME_TEMPORAL_FEATURES = "testBPD3_temporal_features.pkl"
PROCESSED_FILE_NAME_TEMPORAL_PCA = "testBPD3_temporal_pca.pkl"
PROCESSED_FILE_NAME_TEMPORAL_FEATURES_RESP = "testBPD3_temporal_resp.pkl"
PROCESSED_FILE_NAME_TEMPORAL_PCA_RESP = "testBPD3_temporal_resp_pca.pkl"
PROCESSED_FILE_NAME_TEMPORAL_CUT_OFF = "temporal_cut_off_80_98.pkl"
PROCESSED_FILE_NAME_TEMPORAL_PCA_CUT_OFF = "temporal_cut_off_80_98_PCA.pkl"


TEMPORAL_DATA_FOLDER = "temporal4"

MODEL_FOLDER = "models"
METRICS_FOLDER = "metrics"

RANDOM_STATE = 42


# <codecell> Import data


# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython
def load_static(file_location=None, cleaned_data="") -> pd.DataFrame:
    """
    Function to load in different pd.Dataframes
    inputs:
    file_location: path to file to load
    cleaned_data: description of dataframe with locations specified in this file
    """

    if file_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent

        if cleaned_data == "static":
            file_location = (
                project_location / PROCESSED_DATA_FOLDER / PROCESSED_FILE_NAME_STATIC
            )
        elif cleaned_data == "temporal":
            file_location = (
                project_location / PROCESSED_DATA_FOLDER / PROCESSED_FILE_NAME_TEMPORAL
            )
        elif cleaned_data == "temporal_features":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_FEATURES
            )
        elif cleaned_data == "temporal_pca":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_PCA
            )
        elif cleaned_data == "temporal_features_resp":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_FEATURES_RESP
            )
        elif cleaned_data == "temporal_pca_resp":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_PCA_RESP
            )
        elif cleaned_data == "temporal_cut_off":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_CUT_OFF
            )
        elif cleaned_data == "temporal_cut_off_pca":
            file_location = (
                project_location
                / PROCESSED_DATA_FOLDER
                / PROCESSED_FILE_NAME_TEMPORAL_PCA_CUT_OFF
            )
        elif cleaned_data == "spss":
            file_location = project_location / DATA_FOLDER / SSPS_FILE_NAME
        else:
            file_location = project_location / DATA_FOLDER / STATIC_FILE_NAME

    # Change loading function based on suffix
    if file_location.suffix == ".csv":
        # Read csv file and return pandas file with data
        return pd.read_csv(file_location)

    elif file_location.suffix == ".pkl":
        # Read pkl file and return pandas file with data
        with open(file_location, "rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(file_location)


# %%
def save_data(df, data_location):
    """
    Function to save dataframe to pcikle file
    Checks if data file already exists and
    prevents overwriting existing file

    Input:
    df (pd.Dataframe): Pandas dataframe with data to save
    data_location (Union(str,Path)): location of save, can be absolute or relative
    """
    data_location = Path(data_location)
    if not data_location.is_absolute():
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        # Transform data folder to absolute path
        data_location = script_location.parent.parent / data_location

    if data_location.exists():
        print(f"{data_location} already exists, no data saved")
    else:
        with open(data_location, "wb") as f:
            pickle.dump(df, f)
            print(f"Data saved at {data_location}")


# %%
def loadSinglePatient(patient, birthdate, temporal_data_folder=None, time_index=False):
    """
    Function that loads in a pickle of the temporal data of a single patient
    and outputs a pd.Dataframe with all temporal data with a time index
    set to birthdate = 0

    Input:
    patient (str) : ZIS hash of patient
    temporalDataFolder: Path to folder with temporal data
    birthdate (float): Patient birthdate in fraction of day

    Output:
    file_part: pd.DataFrame with temporal data
    newDict: dict with patient data and NP array with temporal data

    """

    # TODO: Update when accepting file at code argument

    if temporal_data_folder is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent
        temporal_data_folder = project_location / DATA_FOLDER / TEMPORAL_DATA_FOLDER

    with open(temporal_data_folder / patient, "rb") as temporal_pickle:
        newDict = pickle.load(temporal_pickle)

    # Create pd.DataFrame with length specified by timeCounter and width based
    # on number of parameters in the dict from the pickle
    # input data is a np.array in the "data" key in the dict sliced on these parameters
    file_part = pd.DataFrame(
        newDict["data"][0 : newDict["timeCounter"], 0 : len(newDict["parNames"])],
        columns=newDict["parNames"],
    )

    # Sort index on timestamps TODO: replace by sorted index on
    # final "Time" Series (kept for compatibility)
    file_part = file_part.sort_values(by="time")
    file_part.reset_index(inplace=True, drop=True)

    # DF can be sorted on pd.TimedeltaIndex, but old version is kept voor compatability
    if time_index:
        # Advantage is that pd.TimedeltaIndex had a build (vectorized=fast!) in round function
        # set time=0 on birthdate in all temporal data
        file_part["Time"] = file_part["time"].index - birthdate
        # Set as pd.timedelta with unit days
        # Replaces division by  (1 / 24 / 60)
        file_part["Time"] = pd.to_timedelta(file_part["Time"], unit="days")
        # Set as index
        file_part = file_part.set_index("Time")
        # Round to nearest milisecond to fix floating point error (replaces roundXSegment)
        file_part.index = file_part.index.round(freq="L")
        # Should be sorted, but can be done again
        # file_part = file_part.sort_index()
        # Consider removing original 'time' column
        # file_part.drop(columns=["time"], inplace = True)

    else:
        # Extract time column and convert to index (has to be an index because the
        # correct rounding function is only implemented on pd.TimedeltaIndex)
        # set time=0 on birthdate in all temporal data
        l = file_part.set_index("time").index - birthdate
        # Replaces division by  (1 / 24 / 60) that changed it to minutes
        l = pd.to_timedelta(l, unit="days")
        # Round to nearest milisecond to fix floating point error (replaces roundXSegment)
        l = l.round(freq="L")
        # Create DF with correct name and remove index
        l = l.to_frame(index=False, name="Time")
        # Join back to original DF
        file_part = file_part.join(l)

    # TODO: review return (better as class or other integrated structure if needed, otherwise not return file_part)
    return file_part, newDict
