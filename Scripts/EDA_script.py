# %%
## Import packages
from Utilities import load_static
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# %%
## Import data
# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython

# Read csv file
data_static = load_static()

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

# %%
data_static.describe()

# %%
data_static.hist(figsize=(10, 10))
plt.show()

# %% [markdown]
# Columns 'smlga' and 'A_5' contain 999 variables, I will recast these as np.nan variables and will look at the histogram again

# %%
display(data_static[data_static["smlga"] == 999])
display(data_static[data_static["A_5"] == 999])

# %%
data_static_cleaned = data_static.replace(999, np.nan)
data_static_cleaned.loc[:, ["smlga", "A_5"]].hist(figsize=(10, 10))

# %%
data_static_cleaned.iloc[[39, 561]]

# %% [markdown]
# Set index and drop columns that are not needed

# %%
# Transform data
# # Set index on hashed identificator
data_static_cleaned = data_static_cleaned.astype({"ZIS": "string"})
data_static_cleaned = data_static_cleaned.set_index("ZIS")

columns_to_drop = ["PT_gb", "Opndat", "Exclusion"]  # , "Furosemide", "DEXA"]
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
# Count missing values (contains 999 codes)
MissingRowCount = data_static_cleaned.isnull().sum().sum()
print(f"Number of rows with missing values: {MissingRowCount}")
print(f"Percentage of total rows {MissingRowCount/data_static_cleaned.shape[0]:%}")

# %% [markdown]
# Because the percentage of missing rows is so low, I will drop these two rows

# %%
data_static_cleaned.dropna(inplace=True)

# %%
CorrelationMatrix = data_static_cleaned.corr()
CorrelationMatrix.style.applymap(
    lambda x: "color:green" if x > 0.6 or x < -0.6 else "color:red"
)

# %% [markdown]
# None of the variables has a correlation with the target greater than 0.4
#
# Only Birthweigth and Gestational Age have a correlation higher than 0.6

# %%
# Save transformed data
utilities_script_location = "Z:/joris.vandervorst/Scripts_Joris/Utilities.py"
data_folder = "processed_data"  # Specify folder name
file_name = "testBPD3_cleaned.pkl"  # Specify file name


def save_data(df, data_location):

    data_location = Path(data_location)
    if not data_location.is_absolute():
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(utilities_script_location)
        )

    # Transform data folder to absolute path
    data_location = script_location.parent.parent / data_location

    if data_location.exists():
        print(f"{data_location} already exists, no data saved")
    else:
        with open(data_location, "wb") as f:
            pickle.dump(df, f)
            print(f"Data saved at {data_location}")


save_data(data_static_cleaned, data_folder + "/" + file_name)
