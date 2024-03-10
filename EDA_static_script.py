# %%
## Import packages
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tableone
from dataloader import (
    PROCESSED_DATA_FOLDER,
    PROCESSED_FILE_NAME_STATIC,
    PROCESSED_FILE_NAME_TEMPORAL,
    load_static,
    loadSinglePatient,
    save_data,
)
from sklearn.linear_model import LogisticRegression
from tqdm.autonotebook import tqdm

# %%
# Settings
main_folder = "Z:\\joris.vandervorst"
data_folder = "data"
temporal_folder = "temporal4"

# Max days we are interested in
DAYS = 7

# %%
## Import data
# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython

# Read csv file
data_static: pd.DataFrame = load_static()

# %%
# Count number of rows and columns
print(f"Static data shape = {data_static.shape}")
# Show column names
print(f" Column names: {data_static.columns}")
# Show data types
print(data_static.dtypes)

# Count missing values (contains 999 codes)
print(f"Number of colums with missing variables: {data_static.isnull().sum().sum()}")

# Check for duplicate rows
print(f"Number of duplicated rows: {data_static.duplicated().sum()}")

# %%
# Show first 5 rows
data_static.head()

# %% [markdown]
# ## Update dataframe with correct birth date
#
# PT_gb is not correct, the correct date of birth will be imported from the "data\static_hashed_SPSS.csv" file

# %%
# Load SPSS files and set index in ZIS hash for comparison
data_SPSS = load_static(cleaned_data="spss").set_index("ZIS")

# %%
# Check for duplicates before updating
data_SPSS[data_SPSS.index.duplicated(keep=False)]

# %% [markdown]
# ### One ZIS hash is duplicated, for now the second one is dropped (first one more filled in)

# %%
# Drop second duplicate hash
data_SPSS = data_SPSS[~data_SPSS.index.duplicated(keep="first")]

# %%
# Only the bithdates are needed to update the original file
spss_birth_dates = data_SPSS["PT_gb"]

data_static_updated = data_static.copy().set_index("ZIS")
data_static_updated.update(spss_birth_dates)

# %%
# Sanity check using pandas merge en LoadPatientStatic function)
main_folder = "Z:\\joris.vandervorst"
import sys

sys.path.append(main_folder)
from ScriptsVanFrank.LoadPatientStatic import getPatients

data_static_Frank = getPatients(target="BPD")

print(
    f"Number of rows and columns with different values: {data_static_updated.compare(data_static_Frank).shape}"
)

# %%
# Show first 5 rows
data_static_updated.head()

# %%
data_static_updated.describe()

# %%
data_static_updated.hist(figsize=(10, 10))
plt.show()

# %% [markdown]
#  Columns 'smlga' and 'A_5' contain 999 variables, I will recast these as np.nan variables and will look at the histogram again

# %%
display(data_static_updated[data_static_updated["smlga"] == 999])
display(data_static_updated[data_static_updated["A_5"] == 999])

# %%
data_static_cleaned = data_static_updated.replace(999, np.nan)
data_static_cleaned.loc[:, ["smlga", "A_5"]].hist(figsize=(10, 10))

# %%
data_static_cleaned.iloc[[39, 561]]

# %% [markdown]
#  Set index and drop columns that are not needed

# %%
# Do not drop PT_gb after update
columns_to_drop = ["Opndat", "Exclusion"]  # , "Furosemide", "DEXA"]
data_static_cleaned = data_static_cleaned.drop(columns_to_drop, axis=1)

# %%
# Count missing values after recasting nan
print(
    f"Number of colums with missing variables: \n{data_static_cleaned.isnull().sum()}"
)

# %%
data_static_cleaned[data_static_cleaned.isna().any(axis=1)]

# %%
data_static_cleaned.describe()

# %%
# Count missing values (contains Na codes)
MissingRowCount = data_static_cleaned.isnull().sum().sum()
print(f"Number of rows with missing values: {MissingRowCount}")
print(f"Percentage of total rows {MissingRowCount/data_static_cleaned.shape[0]:%}")

# %% [markdown]
# ## Imputation of missing values
# There are two columns with missing values: Apgar score at 5 min and small for gestational age. Apgar score will be imputed by the median score. Because small for gestational age is completely dependent om age and birth weigth, a logistic regression will be used to impute this missing value.

# %%
# Impute Apgar score
# Calculate median apgar:
median_Apgar = data_static_cleaned["A_5"].median()
# Impute with median
data_static_cleaned["A_5"].fillna(median_Apgar, inplace=True)

# %%
# Impute small for gestational age
# Create logistic regression model
smlga_LR = LogisticRegression()
Complete_DF = data_static_cleaned.dropna()
smlga_LR.fit(Complete_DF[["GA_exact", "Gebgew"]], y=Complete_DF["smlga"])

# Select row with missing data
Missing_DF = data_static_cleaned[data_static_cleaned["smlga"].isna()]
# Predict missing data
missing_values_smlga = smlga_LR.predict(Missing_DF[["GA_exact", "Gebgew"]])
# Impute missing value
data_static_cleaned.loc[
    data_static_cleaned["smlga"].isna(), "smlga"
] = missing_values_smlga

# %%
CorrelationMatrix = data_static_cleaned.corr()
CorrelationMatrix.style.applymap(
    lambda x: "color:green" if x > 0.6 or x < -0.6 else "color:red"
)

# %% [markdown]
#  None of the variables has a correlation with the target greater than 0.4
#
#  Only Birthweigth and Gestational Age have a correlation higher than 0.6

# %%
# Save transformed data
save_data(data_static_cleaned, PROCESSED_DATA_FOLDER + "/" + PROCESSED_FILE_NAME_STATIC)

# %% [markdown]
# # Filter out patients without Temporal Data
# In order to correctly compare all models, only patients with valid temporal data are included in the dataset

# %%
print(f"Length static dataset {data_static_cleaned.shape[0]}")

# First check if patients have a matching file
data_folder: Path = Path(main_folder) / data_folder
temporal_path: Path = data_folder / temporal_folder
list_temporal = pd.Series([file.name for file in (temporal_path).glob("*")])
list_temporal_matches = data_static_cleaned.index[
    data_static_cleaned.index.isin(list_temporal)
]

print(f"Total number of patient with matching hash file: {len(list_temporal_matches)}")

# Now we are only interested in data of the first 7 days
timeDelta = pd.Timedelta(DAYS, "d")

# List of patient hashes that contain data for the first two weeks
filtered_list_temporal_matches = list()

# List of patient indices that are empty
listEmptyPatients = list()

# Loop over all files and check which patients have valid data
for patient in tqdm(list_temporal_matches):
    birthdate = data_static_cleaned.loc[patient, "PT_gb"]
    timedataDF, _ = loadSinglePatient(patient, birthdate=birthdate)
    # shorten dataset to first DAYS days
    timedataDF = timedataDF[timedataDF["Time"] < timeDelta]

    # Remove all data with negative Time values (recorded before birth)
    timedataDF = timedataDF[timedataDF["Time"] > pd.Timedelta(0, "d")]

    # Do not continue if there is no data for the first two weeks
    if timedataDF.shape[0] == 0:
        listEmptyPatients.append(patient)
        continue

    # Append to list of patients with data
    filtered_list_temporal_matches.append(patient)

print(
    f"Total number of patients with valid data: {len(filtered_list_temporal_matches)}"
)
print(
    f"Number of patients with matching hash but no data in first {DAYS} days: {len(listEmptyPatients)}"
)


# %%
data_static_temporal = data_static_cleaned[
    data_static_cleaned.index.isin(filtered_list_temporal_matches)
]
data_static_temporal.shape

# %%
# Set dytpes of smlga and A_5 (where float because of missing)
data_static_temporal = data_static_temporal.astype({"smlga": "int64", "A_5": "int64"})
data_static_temporal.dtypes

# %%
# Table one

table_renames = {
    "y": "BPD",
    "PT_geslacht": "Sex",
    "GA_exact": "Gestational age (weeks)",
    "Gebgew": "Birth weight (g)",
    "meerlaantal": "Number of births",
    "smlga": "Small for gestational age",
    "ANS": "Antenatal steroid use",
    "A_5": "Apgar score at 5 minutes",
    # "Furosemide": "Postnatal treatment with Furosemide",
    # "DEXA": "Postnatal treatment with Dexamethasone",
}
table_columns = [
    "PT_geslacht",
    "GA_exact",
    "Gebgew",
    "meerlaantal",
    "smlga",
    "ANS",
    "A_5",
    # "Furosemide",
    # "DEXA",
    "y",
]
table_categorical = [
    "PT_geslacht",
    "meerlaantal",
    "smlga",
    "ANS",
    # "Furosemide",
    # "DEXA",
    "y",
]
table_target = "y"
tableOne = tableone.TableOne(
    data_static_temporal,
    columns=table_columns,
    categorical=table_categorical,
    rename=table_renames,
    groupby=table_target,
    missing=False,
    pval=True,
)
tableOne

# %%
tableOne.to_latex("Z:/joris.vandervorst/reports/tableone_p.tex")
tableOne.to_html("Z:/joris.vandervorst/reports/tableone_p.html")

# %%
# Final pairplot
sns.pairplot(
    data_static_temporal.rename(columns={"y": "BPD"}, inplace=False),
    # vars=table_columns,
    height=3,
)
plt.show()

# %%
save_data(
    data_static_temporal, PROCESSED_DATA_FOLDER + "/" + PROCESSED_FILE_NAME_TEMPORAL
)

# %%
