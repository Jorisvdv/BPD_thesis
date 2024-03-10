"""
Function to load in temporal BPD dataset and split into
segments of a certain length, and transform
to torch tensor for use in autoencoder
@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-27

"""

# %%
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataloader import load_static, loadSinglePatient
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from Utils.clean_temporal_data import (
    imputeAndComineFiO2,
    interpolateSpO2,
    removeWrongSpO2,
)
from Utils.extract_temporal_features import invasive_support

# Adjust location of helper files if needed


# %%
# Settings
# TODO: adjust for flexible loading of dataset

RANDOM_SEED = 42
DATA_FOLDER = "data"
TEMPORAL_FOLDER = "temporal4"
MAIN_FOLDER = "Z:\\joris.vandervorst"
DATA_FOLDER: Path = Path(MAIN_FOLDER) / DATA_FOLDER
TEMPORAL_PATH: Path = DATA_FOLDER / TEMPORAL_FOLDER
PROCESSED_DATA_FOLDER = Path(MAIN_FOLDER) / "processed_data"
CACHED_DATA = PROCESSED_DATA_FOLDER / "segment_ends_FiO2_padded.pkl"

# %%
# load in patient list with valid data in first week
df_temporal = load_static(cleaned_data="temporal")


# %%
def extractSegments(
    temporal_stream,
    lengthSegment=120,
    columns=["SpO2"],
    minPerc=None,
):
    # Extract columns and transform into Tensor
    tensor = torch.Tensor(temporal_stream.loc[:, columns].values)

    # Discarding the last rows to make it evenly divisible by chunk_size
    tensor = tensor[: len(tensor) // lengthSegment * lengthSegment]

    # Splitting it into chunks of chunk_size x 2
    chunks = torch.split(tensor, lengthSegment, dim=0)

    # List to keep valid chunks
    valid_chunks = []

    # Iterate through the chunks to filter based on the missing threshold
    for chunk in chunks:
        # Compute the percentage of missing (NaN) values in the chunk
        percentMissing = torch.isnan(chunk).float().mean(dim=0)

        # Only discard segments if a minPercentage is provided
        if minPerc:
            # If the percentage of missing values is more than the threshold, discard the chunk
            if ((1 - percentMissing) <= minPerc).any():
                continue

        # Replace NaN values with 0 in the chunk
        # chunk_no_nan = chunk.masked_fill(torch.isnan(chunk), 0)
        # valid_chunks.append(chunk_no_nan)
        valid_chunks.append(chunk)

    # Return None if there are no valid chunks
    if len(valid_chunks) == 0:
        return None

    # Stack the valid chunks along a new dimension to get a tensor of the desired shape
    stacked_tensor = torch.stack(valid_chunks)

    return stacked_tensor


def loadPatientSegments(
    patient_id,
    patient_birtdate,
    segment_length,
    columns,
    days=7,
    minPerc=None,
    replace_nan=0,
):
    # Load the datafile
    df, _ = loadSinglePatient(patient=patient_id, birthdate=patient_birtdate)

    # Filter out patients with no data before or at days of interest
    maxTime = pd.Timedelta(days, "d")  # adjust for 0 indexing
    df = df[df["Time"] < maxTime]

    # Remove all data with negative Time values (recorded before birth)
    df = df[df["Time"] > pd.Timedelta(0, "d")]

    # Preprocess some SpO2 data
    df = removeWrongSpO2(df)

    # Add data on respiratory status
    df = invasive_support(df)

    # Preprocess FiO2 data
    df = imputeAndComineFiO2(df)

    # Extract segments
    segments = extractSegments(
        temporal_stream=df,
        lengthSegment=segment_length,
        minPerc=minPerc,
        columns=columns,
    )

    # Return None if patient does not have valid data
    if segments is None:
        return None

    # Todo: add interpolation

    if replace_nan is not None:
        # Replace NaN values with 0 in the seqments
        segments = segments.masked_fill(torch.isnan(segments), replace_nan)

    # TODO: Add trained normalizer

    return segments


# %%


def loadAllPatientsSegments(
    staticDF,
    daysToLoad=[0],
    dataFolderPath=None,
    columns=["SpO2"],
    segment_length=120,
    minPerc=None,
    replace_nan=0.0,
    normalize=True,
):
    fileDoesNotExist = list()

    allSegments = list()

    # Filter patients for the existence of a temporal file
    if dataFolderPath is not None:
        file_paths = Path(dataFolderPath).glob("*")
        file_names = [path.stem for path in file_paths]

        # Check if file exist for patients
        index_set = set(staticDF.index)
        files_set = set(file_names)
        matched_patients = index_set & files_set
        unmatched_patients = list(index_set - files_set)

        fileDoesNotExist.extend(unmatched_patients)

        # Filter DataFrame to only include patients with a file present
        staticDF = staticDF[staticDF.index.isin(matched_patients)]

    for patient in tqdm(staticDF[:].iterrows(), total=staticDF.shape[0]):
        segments = loadPatientSegments(
            patient_id=patient[0],
            patient_birtdate=patient[1]["PT_gb"],
            segment_length=segment_length,
            columns=columns,
            # TODO: change so that max days can be directly given
            days=daysToLoad,
            minPerc=minPerc,
            replace_nan=replace_nan,
        )

        # Skip if patient does not have valid data
        # Only applicable if minPerc is given
        if segments is None:
            continue
        if bool((segments == 0).all(dim=1).sum() > 0):
            print(f"Number of empty rows {(segments== 0).all(dim=1).sum().values()}")
        # if (segments== 0).all(dim=1).sum() > 0:
        #     print(patient[0])
        allSegments.append(segments)

    total_tensor = torch.cat(allSegments)
    # print(total_tensor.shape)

    if normalize:
        # Normalize tensor
        tensor_mean = total_tensor.mean(dim=list(range(total_tensor.dim() - 1)))
        tensor_std = total_tensor.std(dim=list(range(total_tensor.dim() - 1)))

        norm_tensor = (total_tensor - tensor_mean) / tensor_std

        return norm_tensor
    else:
        return total_tensor


# %%
def padSegment(segment, target_size, replace_nan=0):
    padding_length = int(target_size - segment.shape[0])
    padded_seq = torch.nn.functional.pad(
        segment, (0, 0, 0, 0, 0, padding_length), "constant", replace_nan
    )
    return padded_seq


def loadAllPatientsPadded(
    staticDF,
    daysToLoad=7,
    dataFolderPath=None,
    columns=["SpO2"],
    segment_length=120,
    minPerc=None,
    replace_nan=0.0,
    normalize=False,
):
    fileDoesNotExist = list()

    allSegments = list()

    # Filter patients for the existence of a temporal file
    if dataFolderPath is not None:
        file_paths = Path(dataFolderPath).glob("*")
        file_names = [path.stem for path in file_paths]

        # Check if file exist for patients
        index_set = set(staticDF.index)
        files_set = set(file_names)
        matched_patients = index_set & files_set
        unmatched_patients = list(index_set - files_set)

        fileDoesNotExist.extend(unmatched_patients)

        # Filter DataFrame to only include patients with a file present
        staticDF = staticDF[staticDF.index.isin(matched_patients)]

    for patient in tqdm(staticDF[:].iterrows(), total=staticDF.shape[0]):
        segments = loadPatientSegments(
            patient_id=patient[0],
            patient_birtdate=patient[1]["PT_gb"],
            segment_length=segment_length,
            columns=columns,
            days=daysToLoad,
            minPerc=minPerc,
            replace_nan=replace_nan,
        )
        # Padd to maximum size
        # subtract 1 from maximum size because all patient have some incomplete time

        # target_size = (daysToLoad * 24 * 60 - 1) / segment_length
        # padding_length = int(target_size - segments.shape[0])
        # print(segments.shape)
        # padded_seq = torch.nn.functional.pad(
        #     segments, (0, 0, 0, 0, 0, padding_length), "constant", replace_nan
        # )
        # padded_seq = padSegment(

        #     segment=segments, target_size=target_size, replace_nan=replace_nan
        # )
        # allSegments.append(padded_seq)
        allSegments.append(segments)
    target_size = max([seq.shape[0] for seq in allSegments])
    allSegments = [
        padSegment(segment=seg, target_size=target_size, replace_nan=replace_nan)
        for seg in allSegments
    ]
    total_tensor = torch.stack(allSegments)

    if normalize:
        # Normalize tensor
        tensor_mean = total_tensor.mean(dim=list(range(total_tensor.dim() - 1)))
        tensor_std = total_tensor.std(dim=list(range(total_tensor.dim() - 1)))

        norm_tensor = (total_tensor - tensor_mean) / tensor_std

        return norm_tensor
    else:
        return total_tensor


# %%
# error_case = ("d0342af8b3d8b8d2", 55)
# All FiO2 before or at 7 days is NaN
