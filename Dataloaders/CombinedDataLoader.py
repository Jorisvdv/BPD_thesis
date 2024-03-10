# %%
# %%
import pickle
from pathlib import Path

from Dataloaders.SegmentsDataloader import (  # df_test,; df_train,; df_val,
    loadAllPatientsPadded,
    loadPatientSegments,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from torch import Tensor, div, nn, sub
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

# %%
MAIN_FOLDER = "Z:\\joris.vandervorst"
DATA_FOLDER = "data"
DATA_FOLDER: Path = Path(MAIN_FOLDER) / DATA_FOLDER
PROCESSED_DATA_FOLDER = Path(MAIN_FOLDER) / "processed_data"
CACHED_DATA = PROCESSED_DATA_FOLDER / "combined_data.pkl"


# %%

input = ["PT_geslacht", "GA_exact", "Gebgew", "meerlaantal", "smlga", "ANS", "A_5"]
target = ["y"]
temp_columns = ["SpO2", "FiO2_filled"]

continuous_columns = ["GA_exact", "Gebgew", "meerlaantal", "A_5"]


class PatientDatasetMem(Dataset):
    def __init__(
        self,
        dataframe,
        input_labels,
        target_label,
        preprocessor=None,
        continuous_columns=["GA_exact", "Gebgew", "meerlaantal", "A_5"],
        segment_length=120,
        daysToLoad=7,
        temp_columns=["SpO2", "FiO2_filled"],
        mask_value=0.0,
        temp_mean=None,
        temp_std=None,
    ):
        self.X = dataframe

        self.y = dataframe.loc[:, target_label]
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = ColumnTransformer(
                transformers=[("num", StandardScaler(), continuous_columns)],
                remainder="passthrough",
            )
            self.preprocessor.fit(self.X.loc[:, input_labels])

        self.X_transformed = self.preprocessor.transform(
            self.X.loc[:, input_labels]
        ).astype("float32")
        self.segmentLength = segment_length
        self.days = daysToLoad
        self.temp_columns = temp_columns
        self.mask_value = mask_value
        self.temp = loadAllPatientsPadded(
            staticDF=self.X,
            daysToLoad=self.days,
            dataFolderPath=None,
            columns=self.temp_columns,
            segment_length=self.segmentLength,
            minPerc=None,
            replace_nan=self.mask_value,
        )
        # Manual preprocessing using standard scaling
        # Get mean and std
        if temp_mean is not None:
            self.temp_mean = temp_mean
        else:
            self.temp_mean = self.temp.mean(dim=list(range(self.temp.dim() - 1)))
        if temp_std is not None:
            self.temp_std = temp_std
        else:
            self.temp_std = self.temp.std(dim=list(range(self.temp.dim() - 1)))
        # set flag such that scaling only happens once
        self.scaled = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_transformed = self.X_transformed[index]
        label = self.y.iloc[index].values
        # Get temporal sequences
        temporal_sequences = self.temp[index]

        return (x_transformed, temporal_sequences, label)

    def scale_temp(self):
        if not self.scaled:
            # transform temporal data
            self.temp = (self.temp - self.temp_mean) / self.temp_std
            self.scaled = True
        else:
            print("Data already scaled")

    def unscale_temp(self, tensor):
        # Transform tensor back to orignal values for display
        rescaled = (tensor * self.temp_std) + self.temp_mean
        return rescaled


# %%
# Create custom datasets for training, validation, and test sets

# train = PatientDataset(df_train, input, target, preprocessor)
# val = PatientDataset(df_val, input, target, preprocessor)
# test = PatientDataset(df_test, input, target, preprocessor)
# # %%
# # Create DataLoaders for training and validation sets
# train_dataloader = DataLoader(train, batch_size=30, shuffle=True)
# val_dataloader = DataLoader(val, batch_size=30)
# test_dataloader = DataLoader(test, batch_size=30)

# %%
# x1, temp1, y1 = next(iter(train_dataloader))
# x2, temp2, y2 = next(iter(train_dataloader))
# print(temp1.shape)
# print(temp2.shape)
# print(len(x2))

# %%
# train_mem = PatientDatasetMem(df_train, input, target, preprocessor)
# val_mem = PatientDatasetMem(df_val, input, target, preprocessor)
# test_mem = PatientDatasetMem(df_test, input, target, preprocessor)
# %%
# train_mem = PatientDatasetMem(df_train, input, target)
# val_mem = PatientDatasetMem(
#     df_val,
#     input,
#     target,
#     preprocessor=train_mem.preprocessor,
#     temp_mean=train_mem.temp_mean,
#     temp_std=train_mem.temp_std,
# )
# test_mem = PatientDatasetMem(
#     df_test,
#     input,
#     target,
#     preprocessor=train_mem.preprocessor,
#     temp_mean=train_mem.temp_mean,
#     temp_std=train_mem.temp_std,
# )
# # %%
# train_dataloader_mem = DataLoader(train_mem, batch_size=30, shuffle=True)
# val_dataloader_mem = DataLoader(val_mem, batch_size=30)
# test_dataloader_mem = DataLoader(test_mem, batch_size=30)
# %%
# if CACHED_DATA.exists():
#     # if False:
#     with open(CACHED_DATA, "rb") as f:
#         dataset = pickle.load(f)
#         dataset_train = dataset["train"]
#         dataset_val = dataset["val"]
#         dataset_test = dataset["test"]
#         dataset_train_SpO2 = dataset["train_SpO2"]
#         dataset_val_SpO2 = dataset["val_SpO2"]
#         dataset_test_SpO2 = dataset["test_SpO2"]
# # if False: pass

# else:
#     dataset_train = PatientDatasetMem(df_train, input, target)
#     dataset_val = PatientDatasetMem(
#         df_val,
#         input,
#         target,
#         preprocessor=dataset_train.preprocessor,
#         temp_mean=dataset_train.temp_mean,
#         temp_std=dataset_train.temp_std,
#     )
#     dataset_test = PatientDatasetMem(
#         df_test,
#         input,
#         target,
#         preprocessor=dataset_train.preprocessor,
#         temp_mean=dataset_train.temp_mean,
#         temp_std=dataset_train.temp_std,
#     )
#     dataset_train_SpO2 = PatientDatasetMem(
#         df_train, input, target, temp_columns=["SpO2"]
#     )
#     dataset_val_SpO2 = PatientDatasetMem(
#         df_val,
#         input,
#         target,
#         temp_columns=["SpO2"],
#         preprocessor=dataset_train_SpO2.preprocessor,
#         temp_mean=dataset_train_SpO2.temp_mean,
#         temp_std=dataset_train_SpO2.temp_std,
#     )
#     dataset_test_SpO2 = PatientDatasetMem(
#         df_test,
#         input,
#         target,
#         temp_columns=["SpO2"],
#         preprocessor=dataset_train_SpO2.preprocessor,
#         temp_mean=dataset_train_SpO2.temp_mean,
#         temp_std=dataset_train_SpO2.temp_std,
#     )

#     # Scale temporal data
#     dataset_train.scale_temp()
#     dataset_val.scale_temp()
#     dataset_test.scale_temp()
#     dataset_train_SpO2.scale_temp()
#     dataset_val_SpO2.scale_temp()
#     dataset_test_SpO2.scale_temp()

#     with open(CACHED_DATA, "wb") as f:
#         dataset = dict()
#         dataset["train"] = dataset_train
#         dataset["val"] = dataset_val
#         dataset["test"] = dataset_test
#         dataset["train_SpO2"] = dataset_train_SpO2
#         dataset["val_SpO2"] = dataset_val_SpO2
#         dataset["test_SpO2"] = dataset_test_SpO2
#         pickle.dump(dataset, f)
#         print("file_saved")

# %%
