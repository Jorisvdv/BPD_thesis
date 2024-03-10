"""
Script to extract features for temporal data

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-19

"""
# %% Load dependencies
# Dependencies
from pathlib import Path

import numpy as np
import pandas as pd
from dataloader import (
    PROCESSED_DATA_FOLDER,
    PROCESSED_FILE_NAME_TEMPORAL_CUT_OFF,
    PROCESSED_FILE_NAME_TEMPORAL_FEATURES,
    PROCESSED_FILE_NAME_TEMPORAL_FEATURES_RESP,
    PROCESSED_FILE_NAME_TEMPORAL_PCA,
    PROCESSED_FILE_NAME_TEMPORAL_PCA_CUT_OFF,
    PROCESSED_FILE_NAME_TEMPORAL_PCA_RESP,
    load_static,
    loadSinglePatient,
    save_data,
)
from tqdm.notebook import tqdm
from Utils.clean_temporal_data import (
    imputeAndComineFiO2,
    interpolateSpO2,
    removeWrongSpO2,
)
from Utils.extract_temporal_features import (
    SpO2FiO2dev,
    createSummaryFun,
    hyperoxia,
    hypoxia,
    invasive_support,
)

# %% Settings
# Settings
main_folder = "Z:\\joris.vandervorst"
data_folder = Path("data")
temporal_folder = "temporal4"
DAYS = 7
MIN_PRESENT = 0.1

data_folder: Path = Path(main_folder) / data_folder
temporal_path: Path = data_folder / temporal_folder

# %% Load static patient data
data_static = load_static(cleaned_data="temporal")


# %% Extract temporal features

extFeatList = list()

# Filter to shorten dataset to first DAYS
timeDelta = pd.Timedelta(value=DAYS, unit="d")

# Instantiate summary functions
# Ensure returning None when not enough data
colmean = createSummaryFun(pd.Series.mean, min_present=MIN_PRESENT)
colstd = createSummaryFun(pd.Series.std, min_present=MIN_PRESENT)
colvar = createSummaryFun(pd.Series.var, min_present=MIN_PRESENT)

hypox = createSummaryFun(hypoxia, min_present=MIN_PRESENT)
hyperox = createSummaryFun(hyperoxia, min_present=MIN_PRESENT)

print("Create dataframe with extracted temporal features")
# Loop over list of patients
for patient in tqdm(data_static.index):
    birthdate = data_static.loc[patient, "PT_gb"]

    # Load in patient temporal data
    timedataDF, _ = loadSinglePatient(patient=patient, birthdate=birthdate)

    # shorten dataset to first DAYS days
    timedataDF = timedataDF[timedataDF["Time"] < timeDelta]

    # Remove all data with negative Time values (recorded before birth)
    timedataDF = timedataDF[timedataDF["Time"] > pd.Timedelta(value=0, unit="d")]

    # Clean data and create featues
    timedataDF = removeWrongSpO2(timedataDF)
    timedataDF = interpolateSpO2(timedataDF)

    # To add type of respiratory support based on column names uncomment next line
    # timedataDF = invasive_support(timedataDF)

    timedataDF = imputeAndComineFiO2(timedataDF, print_combinations=False)
    timedataDF = SpO2FiO2dev(timedataDF)

    # Create days column to group by
    timedataDF["days"] = timedataDF["Time"].dt.days

    # Calculate summary features (mean and variance) per day
    result = timedataDF.groupby("days")[
        ["SpO2_filled", "FiO2_filled", "SpO2FiO2dev"]
    ].agg([colmean, colvar])

    # Transform multi-index with sumaary statistics to dict with featues
    result_dict = dict()
    for colnames, series in result.items():
        feature, summary = colnames
        for i, v in zip(series.index, series.values):
            result_dict[f"{feature}_{summary}_{i+1:02}"] = v

    # Create hypoxia and hyperoxia features
    # Calculate per day
    hypoxic = timedataDF.groupby("days").apply(hypoxia, cut_off=90)
    hyperoxic = timedataDF.groupby("days").apply(hyperoxia, cut_off=95)
    # Add to dict
    for day in hypoxic.index:
        result_dict[f"hypoxic_burden_{day+1:02}"] = hypoxic[day]
    for day in hyperoxic.index:
        result_dict[f"hyperoxic_burden_{day+1:02}"] = hyperoxic[day]

    ZIS = {"ZIS": patient}

    # Add ZIS to assign results to patient
    totalExFeat = ZIS | result_dict

    # Add patient to list with all patients
    extFeatList.append(totalExFeat)

# Transform list of dicts with patient data to dataframe
extFeatDF: pd.DataFrame = pd.DataFrame(extFeatList).set_index("ZIS").sort_index(axis=1)

# %%
# Merge with static data
DF_temporal_features = data_static.join(extFeatDF, how="inner")
# Print list of all columns
print("\n".join(f"- {i}" for i in DF_temporal_features.columns))


# %%
save_data(
    DF_temporal_features,
    PROCESSED_DATA_FOLDER + "/" + PROCESSED_FILE_NAME_TEMPORAL_FEATURES,
)

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# %%
# PCA Transformation
from sklearn.preprocessing import StandardScaler

# %%

feature_names = [
    "FiO2_filled_mean",
    "FiO2_filled_var",
    "SpO2FiO2dev_mean",
    "SpO2FiO2dev_var",
    "SpO2_filled_mean",
    "SpO2_filled_var",
    "hyperoxic_burden",
    "hypoxic_burden",
    # "resp_support_mean",
    # "resp_support_var",
]

# Merge all days in order to get a PCA that can be used for every day
df_list = []
for day in range(1, 8):
    selected_columns = [col for col in extFeatDF.columns if col[-2:] == f"0{day}"]
    df_day = extFeatDF[selected_columns]
    df_day.columns = feature_names
    df_list.append(df_day)

df_merged = pd.concat(df_list, axis=0, ignore_index=True)

# %%
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_merged)

imputer = SimpleImputer(strategy="median")
df_scaled_imputed = imputer.fit_transform(df_scaled)

pca = PCA()
pca_result = pca.fit_transform(df_scaled_imputed)

# Create Scree plot
explained_var = pca.explained_variance_ratio_ * 100
plt.figure()
plt.plot(
    range(1, len(explained_var) + 1),
    explained_var,
    # align="center",
    label="Individual explained variance",
)
plt.plot(
    range(1, len(explained_var) + 1),
    np.cumsum(explained_var),
    label="Cumlative explained varaiance",
)
plt.xlabel("Principle Component Index")
plt.ylabel("Explained Variance Ratio (%)")
plt.title(f"Scree plot temporal featues")
plt.legend(loc="best")
plt.tight_layout()
plt.show

# %%
# 2 PC selected based on scree plot
components_to_keep = 2
pca_final = PCA(n_components=components_to_keep)
pca_final.fit(df_scaled_imputed)

# %%
df_pca_combined = pd.DataFrame()
# Apply PCA and transformations for each day
for day in range(1, 8):
    selected_columns = [col for col in extFeatDF.columns if col[-2:] == f"0{day}"]
    df_day = extFeatDF[selected_columns]
    df_day.columns = feature_names
    df_day_scaled_imputed = imputer.transform(scaler.transform(df_day))
    pca_result_day = pca_final.transform(df_day_scaled_imputed)

    # Store result in DF
    day_columns = [f"Day{day}_PC{i}" for i in range(components_to_keep)]
    df_pca_day = pd.DataFrame(pca_result_day, columns=day_columns)
    df_pca_combined = pd.concat([df_pca_combined, df_pca_day], axis=1)

# %%
# Add index back to PCA transformed data
df_pca_combined.index = extFeatDF.index
# Merge with static data
DF_temporal_PCA = data_static.join(df_pca_combined, how="inner")
# Print list of all columns
print("\n".join(f"- {i}" for i in DF_temporal_PCA.columns))


# %%
save_data(
    DF_temporal_PCA,
    PROCESSED_DATA_FOLDER + "/" + PROCESSED_FILE_NAME_TEMPORAL_PCA,
)

# %%
