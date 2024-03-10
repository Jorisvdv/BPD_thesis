"""
Training loop for LSTM Autoencoder

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
from Dataloaders.SegmentsDataloader import loadAllPatientsSegments
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from Models.AE_LSTM import RecurrentAutoencoder
from torch.utils.data import DataLoader
from Utils.recentCheckpoint import getRecentCheckpoint

# %%
logger_path = Path("Z:\joris.vandervorst\logs")

L.seed_everything(42, workers=True)


# %%
# Uncomment this cell to directly load segments instead of using chached data

# output_train_SpO2 = loadAllPatientsSegments(
#     df_train,
#     daysToLoad=7,
#     columns=["SpO2"],
#     minPerc=0.99,
# )
# output_val_SpO2 = loadAllPatientsSegments(
#     df_val,
#     daysToLoad=7,
#     columns=["SpO2"],
#     minPerc=0.99,
# )

# %%
# Load data into dataloader

# 1 dims dataloader
train_dataloader = DataLoader(
    output_train_SpO2,
    batch_size=500,
    shuffle=True,
)
val_dataloader = DataLoader(output_val_SpO2, batch_size=500, shuffle=False)


# %%
# Define model

model = RecurrentAutoencoder(seq_len=120, n_features=1, embedding_dim=16, loss="L1")


# %%
# Settings for Lightning trainer
model_name = "LSTM_AE_SpO2"
run_name = "16_L1_pretrained"
TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    model_name="LSTM_AE_SpO2",
    version=run_name,
    # sub_dir=,
    log_graph=True,
)
# Save best epoch and stop training after no improvement for 50 epochs
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)
# %%
# Load model weights using checkpoint


# latest_checkpoint = getRecentCheckpoint(
#     r"Z:\joris.vandervorst\logs\LSTM_AE_SpO2\" + run_name
# )
# model.load_from_checkpoint(latest_checkpoint)

# model_dict = torch.load(latest_checkpoint)
# # model.load_state_dict(model_dict["state_dict"])
# model = RecurrentAutoencoder.load_from_checkpoint(
#     latest_checkpoint, **model_dict["hyper_parameters"]
# )
# Load weigths
pretrained_weights = (
    r"Z:\joris.vandervorst\Scripts_Joris\Models\autoencoderFrankWeights.pth"
)
model.load_state_dict(torch.load(pretrained_weights))


# %%
# Instantiate trainer
if __name__ == "__main__":
    trainer = L.Trainer(
        max_epochs=1000,
        logger=[TBLogger],
        log_every_n_steps=5,
        callbacks=[checkpoint, earlystopping],
        # fast_dev_run=True,  # Uncomment for quick loop of 1 epoch without logs
        max_time={"hours": 11}
        # gradient_clip_val=1,
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
