"""
Helper functions for calculating daily summary statistics for temporal data

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09

"""


from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

# %% Summary creator missing data
# Create a wrapper function that returns a summary calculator that first checks data length


def createSummaryFun(summaryFun, min_present=0.1):
    @wraps(summaryFun)
    def summary(DFcolumns):
        # set Column as na if to much missing values
        # Check length of missing features
        percentFilled = DFcolumns.notna().sum() / len(DFcolumns)
        # Do not calculate statistic if non enough present
        if (percentFilled < min_present).all():
            return None
        else:
            return summaryFun(DFcolumns)

    return summary


# %% Create aggrigation functions to calculate hypo-, hyperoxemia
def hypoxia(df, cut_off=90):
    cols = ["SpO2_filled"]
    # Check if both columns are in the dataframe otherwise pass
    if not all(col in df.columns for col in cols):
        return None

    hypoxiaCount = np.where(df["SpO2_filled"] < cut_off, cut_off - df["SpO2_filled"], 0)

    return hypoxiaCount.sum() / len(df)

    # return featureDict


def hyperoxia(df, cut_off=95):
    cols = ["FiO2_filled", "SpO2_filled"]
    # Check if both columns are in the dataframe otherwise pass
    if not all(col in df.columns for col in cols):
        return None

    # Only count if FiO2 > 21
    hyperoxiaCount = np.where(
        (df["SpO2_filled"] > cut_off) & (df["FiO2_filled"] > 21),
        df["SpO2_filled"] - cut_off,
        0,
    )
    return hyperoxiaCount.sum() / len(df)


# %% Create function to calculate SpO2/FiO2 is possible
def SpO2FiO2dev(df):
    cols = ["FiO2_filled", "SpO2_filled"]
    # Check if both columns are in the dataframe otherwise pass
    if all(col in df.columns for col in cols):
        # Add SpO2FiO2dev column
        df["SpO2FiO2dev"] = df["SpO2_filled"] / df["FiO2_filled"]
    else:
        df["SpO2FiO2dev"] = np.nan  # otherwise add empty column

    return df


def invasive_support(df, imputelimit=180):
    """
    Function to create a categorial value to indicate respiratory support
    Coding will be as follows:
    2 when there is a non-zero value above 21 in "FiO2"
    1 when there is a non-zero-value above 21 in "FiO2 onbeademd"
    0 otherwise
    """
    # Create na-filled column
    df.loc[:, "resp_support"] = np.nan

    if "FiO2" in df.columns:
        df.loc[(df["FiO2"] > 21), "resp_support"] = 2
        # Sometimes FiO2 is set as 0.0 when no support
        df.loc[(df["FiO2"] < 1), "resp_support"] = 0
    if "FiO2 onbeademd" in df.columns:
        df.loc[(df["FiO2 onbeademd"] > 21), "resp_support"] = 1
        # Sometimes FiO2 is set as 0.0 when no support
        df.loc[(df["FiO2 onbeademd"] < 1), "resp_support"] = 0
    # Forward fill missing values up to 180 minutes
    # May result in a slight overestimation of resporatory support by max 180 minutes
    df.loc[:, "resp_support"] = df["resp_support"].fillna(
        method="ffill", limit=imputelimit
    )
    # Replace remaining values with 0
    df["resp_support"] = df["resp_support"].fillna(0)

    # Recast as int
    df["resp_support"] = df["resp_support"].astype("int")

    return df
