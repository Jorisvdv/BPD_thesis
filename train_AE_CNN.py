"""
Training loop for CNN Autoencoder

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-13
"""

# %%

from pathlib import Path

import lightning as L
import torch
from cached_data import (
    df_train,
    df_val,
    output_train,
    output_train_SpO2,
    output_val,
    output_val_SpO2,
)
from Dataloaders.CombinedDataLoader import (
    PatientDatasetMem,
    input,
    target,
    temp_columns,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from Models.AE_CNN import CNNAutoencoder, CNNAutoencoderPooled
from torch.utils.data import DataLoader
from Utils.recentCheckpoint import getRecentCheckpoint

# %%

L.seed_everything(42, workers=True)


# %%
# # Uncomment cell to load data manually

# # temp_columns.append("resp_support")
# dataset_train = PatientDatasetMem(
#     df_train,
#     input,
#     target,
#     temp_columns=["SpO2"],
#     segment_length=120,
# )
# dataset_val = PatientDatasetMem(
#     df_val,
#     input,
#     target,
#     segment_length=120,
#     temp_columns=["SpO2"],
#     preprocessor=dataset_train.preprocessor,
#     temp_mean=dataset_train.temp_mean,
#     temp_std=dataset_train.temp_std,
# )

# dataset_train.scale_temp()
# dataset_val.scale_temp()

# # Shift segments per patient in order to increase the number of training segments

# def shift_segments(dataset):
#     total_list = []
#     # Loop over all segments
#     for i in range(0, dataset.temp.shape[2]):
#         # First merge all segments, discard i timesteps and
#         # adjust shape to original segment length
#         total_list.append(
#             dataset.temp.flatten(start_dim=1, end_dim=2)[
#                 :, i : -(dataset.temp.shape[2] - i), :
#             ].view(
#                 dataset.temp.shape[0],
#                 dataset.temp.shape[1] - 1,
#                 dataset.temp.shape[2],
#                 dataset.temp.shape[3],
#             )
#         )
#     return torch.cat(total_list)

# dataset_train_shifted = shift_segments(dataset_train)
# dataset_val_shifted = shift_segments(dataset_val)

# shapes = dataset_train_shifted.shape
# output_train = dataset_train_shifted.view(shapes[0] * shapes[1], shapes[2], shapes[3])

# shapes = dataset_val_shifted.shape
# output_val = dataset_val_shifted.view(shapes[0] * shapes[1], shapes[2], shapes[3])

# %%
# Load data into dataloader
train_dataloader = DataLoader(
    output_train,
    batch_size=5000,
    shuffle=True,
)
val_dataloader = DataLoader(
    output_val,
    batch_size=5000,
    shuffle=False,
)

# %%
# Define model

model = CNNAutoencoder(
    n_features=2,
    seq_len=120,
    latent_size=32,
    loss="MSE",
)


# %%
# Settings for Lightning trainer
logger_path = Path("Z:\joris.vandervorst\logs")
model_name = "CNN_AE_SpO2_FiO2"
run_name = "32_MSE_shifted"
TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name=model_name,
    version=run_name,
    # log_graph=True,
)
# Save best epoch and stop training after no improvement for 50 epochs
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)
# %%
# Instantiate trainer
if __name__ == "__main__":
    trainer = L.Trainer(
        max_epochs=10000,
        logger=[TBLogger],
        log_every_n_steps=1,
        callbacks=[checkpoint, earlystopping],
        # fast_dev_run=True,  # Uncomment for quick loop of 1 epoch without logs
        max_time={"hours": 60},
        # default_root_dir=logger_path,
    )


# %%
# Start training
if __name__ == "__main__":
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        # ckpt_path=latest_checkpoint,
    )

# %%
