"""
File to pre-process data and save cached as pickle
Also stores train, validation, test split

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09

"""

# %%
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataloader import load_static, loadSinglePatient
from Dataloaders.CombinedDataLoader import PatientDatasetMem, input, target
from Dataloaders.SegmentsDataloader import (
    loadAllPatientsPadded,
    loadAllPatientsSegments,
    loadPatientSegments,
)
from sklearn.model_selection import train_test_split

# %%
MAIN_FOLDER = "Z:\\joris.vandervorst"
DATA_FOLDER = "data"
DATA_FOLDER: Path = Path(MAIN_FOLDER) / DATA_FOLDER
PROCESSED_DATA_FOLDER = Path(MAIN_FOLDER) / "processed_data"
CACHED_DATASET = PROCESSED_DATA_FOLDER / "combined_data.pkl"
CACHED_SEGEMENTS = PROCESSED_DATA_FOLDER / "segment_ends_FiO2_padded.pkl"
TEMPORAL_FOLDER = "temporal4"
TEMPORAL_PATH: Path = DATA_FOLDER / TEMPORAL_FOLDER

RANDOM_SEED = 42

# %%
# load in patient list with valid data in first week
df_temporal = load_static(cleaned_data="temporal")

# Load segments for AE
if CACHED_SEGEMENTS.exists():
    # if False:
    with open(CACHED_SEGEMENTS, "rb") as f:
        output = pickle.load(f)
        output_train = output["train"]
        output_val = output["val"]
        output_test = output["test"]
        output_train_SpO2 = output["train_SpO2"]
        output_val_SpO2 = output["val_SpO2"]
        output_test_SpO2 = output["test_SpO2"]
        df_test = output["df_test"]
        df_val = output["df_val"]
        df_train = output["df_train"]
# if False: pass

else:
    df_temp, df_test = train_test_split(
        df_temporal[:],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df_temporal.loc[:, "y"],
    )
    df_train, df_val = train_test_split(
        df_temp[:],
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df_temp.loc[:, "y"],
    )
    output_train = loadAllPatientsSegments(
        df_train,
        daysToLoad=7,
        columns=["SpO2", "FiO2_filled"],
        minPerc=0.2,
    )
    output_val = loadAllPatientsSegments(
        df_val,
        daysToLoad=7,
        columns=["SpO2", "FiO2_filled"],
        minPerc=0.2,
    )
    output_test = loadAllPatientsSegments(
        df_test,
        daysToLoad=7,
        columns=["SpO2", "FiO2_filled"],
        minPerc=0.2,
    )

    output_train_SpO2 = loadAllPatientsSegments(
        df_train,
        daysToLoad=7,
        columns=["SpO2"],
        minPerc=0.2,
    )
    output_val_SpO2 = loadAllPatientsSegments(
        df_val,
        daysToLoad=7,
        columns=["SpO2"],
        minPerc=0.2,
    )
    output_test_SpO2 = loadAllPatientsSegments(
        df_test,
        daysToLoad=7,
        columns=["SpO2"],
        minPerc=0.2,
    )

    with open(CACHED_SEGEMENTS, "wb") as f:
        output = dict()
        output["df_train"] = df_train
        output["df_val"] = df_val
        output["df_test"] = df_test
        output["train"] = output_train
        output["val"] = output_val
        output["test"] = output_test
        output["train_SpO2"] = output_train_SpO2
        output["val_SpO2"] = output_val_SpO2
        output["test_SpO2"] = output_test_SpO2
        pickle.dump(output, f)
        print("file_saved")


# %%
if CACHED_DATASET.exists():
    # if False:
    with open(CACHED_DATASET, "rb") as f:
        dataset = pickle.load(f)
        dataset_train = dataset["train"]
        dataset_val = dataset["val"]
        dataset_test = dataset["test"]
        dataset_train_SpO2 = dataset["train_SpO2"]
        dataset_val_SpO2 = dataset["val_SpO2"]
        dataset_test_SpO2 = dataset["test_SpO2"]
# if False: pass

else:
    dataset_train = PatientDatasetMem(df_train, input, target)
    dataset_val = PatientDatasetMem(
        df_val,
        input,
        target,
        preprocessor=dataset_train.preprocessor,
        temp_mean=dataset_train.temp_mean,
        temp_std=dataset_train.temp_std,
    )
    dataset_test = PatientDatasetMem(
        df_test,
        input,
        target,
        preprocessor=dataset_train.preprocessor,
        temp_mean=dataset_train.temp_mean,
        temp_std=dataset_train.temp_std,
    )
    dataset_train_SpO2 = PatientDatasetMem(
        df_train, input, target, temp_columns=["SpO2"]
    )
    dataset_val_SpO2 = PatientDatasetMem(
        df_val,
        input,
        target,
        temp_columns=["SpO2"],
        preprocessor=dataset_train_SpO2.preprocessor,
        temp_mean=dataset_train_SpO2.temp_mean,
        temp_std=dataset_train_SpO2.temp_std,
    )
    dataset_test_SpO2 = PatientDatasetMem(
        df_test,
        input,
        target,
        temp_columns=["SpO2"],
        preprocessor=dataset_train_SpO2.preprocessor,
        temp_mean=dataset_train_SpO2.temp_mean,
        temp_std=dataset_train_SpO2.temp_std,
    )

    # Scale temporal data
    dataset_train.scale_temp()
    dataset_val.scale_temp()
    dataset_test.scale_temp()
    dataset_train_SpO2.scale_temp()
    dataset_val_SpO2.scale_temp()
    dataset_test_SpO2.scale_temp()

    with open(CACHED_DATASET, "wb") as f:
        dataset = dict()
        dataset["train"] = dataset_train
        dataset["val"] = dataset_val
        dataset["test"] = dataset_test
        dataset["train_SpO2"] = dataset_train_SpO2
        dataset["val_SpO2"] = dataset_val_SpO2
        dataset["test_SpO2"] = dataset_test_SpO2
        pickle.dump(dataset, f)
        print("file_saved")

# %%
