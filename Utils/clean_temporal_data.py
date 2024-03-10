"""
Helper functions for cleaning temporal data

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09

"""

import numpy as np
import pandas as pd


def imputeAndComineFiO2(df, print_combinations=False, imputelimit=180):
    if print_combinations:
        cols = ["FiO2", "FiO2 onbeademd"]
        # Check if both columns are in the dataframe otherwise pass
        if all(col in df.columns for col in cols):
            # Identify cases where both FiO2 and FiO2 onbeademd have valid values
            valid_both = df.dropna(subset=["FiO2", "FiO2 onbeademd"])
            if not valid_both.empty:
                print("Rows containing both valid 'FiO2' and 'FiO2 onbeademd' values")
                print(valid_both[["Time", "FiO2", "FiO2 onbeademd"]])

            # values_within_5_min_list = list()

            # for i in range(1, 6):
            #     mask = df["FiO2"].notna() & df["FiO2 onbeademd"].shift(i).notna()
            #     values_within_5_min_list.append(df.loc[mask])

            # for i in range(1, 6):
            #     mask = df["FiO2"].shift(i).notna() & df["FiO2 onbeademd"].notna()
            #     values_within_5_min_list.append(df.loc[mask])

            # values_within_5_min = pd.concat(values_within_5_min_list).drop_duplicates()

            # if not values_within_5_min.empty:
            #     print(
            #         "Rows with valid 'FiO2' and 'FiO2 onbeademd' values within 5 minutes"
            #     )
            #     print(values_within_5_min[["Time", "FiO2", "FiO2 onbeademd"]])

    # Combine 'FiO2', 'FiO2 onbeademd' columns
    if "FiO2" in df.columns and "FiO2 onbeademd" in df.columns:
        # Sometimes FiO2 is set as 0.0 when transfered no support, remove those values
        df.loc[:, "FiO2_filled"] = df["FiO2"].replace(0.0, np.nan)
        df.loc[:, "FiO2_filled"] = df["FiO2_filled"].combine_first(df["FiO2 onbeademd"])
    elif "FiO2" in df.columns:
        # Sometimes FiO2 is set as 0.0 when transfered no support, remove those values
        df.loc[:, "FiO2_filled"] = df["FiO2"].replace(0.0, np.nan)
    elif "FiO2 onbeademd" in df.columns:
        df.loc[:, "FiO2_filled"] = df["FiO2 onbeademd"]
    else:
        df.loc[:, "FiO2_filled"] = np.nan

    # Forward fill missing values up to 180 minutes
    df.loc[:, "FiO2_filled"] = df["FiO2_filled"].fillna(
        method="ffill", limit=imputelimit
    )

    return df


def removeWrongSpO2(df):
    spo2_ind = list(df.columns).index("SpO2")

    # Where deviations between SpO2 and ECG > 10, fill in with NaNs
    # If HF or Pulse is not present, it is assumed right
    wrongSpo2 = np.where(np.abs(df["HF"] - df["Pulse"]) > 10)[0]
    df.iloc[wrongSpo2, spo2_ind] = np.nan

    # Where SpO2 < 50, fill in with 50
    SpO2_low = np.where(df["SpO2"] < 50)[0]
    df.iloc[SpO2_low, spo2_ind] = 50
    return df


def interpolateSpO2(df, min_present=0):
    len_df = len(df)
    len_notNaN = len(np.where(~np.isnan(df["SpO2"]))[0])

    if (len_notNaN / len_df) <= min_present:
        df["SpO2_filled"] = np.nan
        return df

    df["SpO2_filled"] = df["SpO2"].interpolate(method="linear")

    isEmpty = np.where(np.isnan(df["SpO2"]))[0]
    tofill = np.random.normal(0, 1, len(isEmpty))

    df.iloc[isEmpty, -1] = df.iloc[isEmpty, -1] + tofill
    tooHigh = np.where(df.iloc[:, -1] > 100)[0]
    df.iloc[tooHigh, -1] = 100

    return df
