"""
Training loop for LSTM Clasifier using LSTM as autoencoder
This version includes information on the fix of the AE model using it's loss function

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""
# %%
from pathlib import Path

import lightning as L
import torch
from cached_data import dataset_train_SpO2, dataset_val_SpO2, df_test, df_train, df_val
from Dataloaders.CombinedDataLoader import (  # dataset_train_SpO2,; dataset_val_SpO2,
    PatientDatasetMem,
    input,
    target,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from Utils.recentCheckpoint import getRecentCheckpoint

# %%
L.seed_everything(42, workers=True)


# %%
# Load model architechture
from Models.AE_LSTM import RecurrentAutoencoderWithLoss
from Models.LSTMClasifier import LSTMClassifierWithLoss

# %%
# Load in AE weights for encoder

CHECKPOINT_encoder = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\LSTM_AE_SpO2\old_pretrained_16_l1"
)

# Instantiate Encoder
model_AE = RecurrentAutoencoderWithLoss.load_from_checkpoint(CHECKPOINT_encoder)

# Freeze encoder weights
for param in model_AE.parameters():
    param.requires_grad = False

# %%

model = LSTMClassifierWithLoss(
    input_size=1,
    embedding_size=model_AE.latent_size,
    hidden_size=64,
    num_layers=8,
    dropout_perc=0.25,
    Aencoder=model_AE,
    lr=1e-3,
)
# %%
# Load in dataset

train_dataloader = DataLoader(dataset_train_SpO2, batch_size=60, shuffle=True)
val_dataloader = DataLoader(dataset_val_SpO2, batch_size=60)


# %%
# Settings for Lightning trainer
logger_path = Path(r"Z:/joris.vandervorst/logs")
model_name = "LSTM_Clasifier_SpO2"
run_name = f"load{train_dataloader.batch_size}_clip_1_{model.lstm.hidden_size}_{model.lstm.num_layers}_pretrained_16_l1"

TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name=model_name,
    version=run_name,
    # sub_dir=,
    log_graph=True,
)
# Save best epoch and stop training after no improvement for 50 epochs
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)
# %%
# Create a Trainer and train the model
trainer = L.Trainer(
    max_epochs=10000,
    logger=[TBLogger],
    log_every_n_steps=5,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,  # Uncomment for quick loop of 1 epoch without logs
    max_time={"hours": 24, "minutes": 0},
    # accumulate_grad_batches=30,
    gradient_clip_val=1,
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
