"""
Training loop for LSTM Clasifier using CNN as autoencoder

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09
"""

# %%
from pathlib import Path

import lightning as L
import torch
from cached_data import (
    dataset_train,
    dataset_train_SpO2,
    dataset_val,
    dataset_val_SpO2,
    df_test,
    df_train,
    df_val,
)
from Dataloaders.CombinedDataLoader import PatientDatasetMem, input, target
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from Utils.recentCheckpoint import getRecentCheckpoint

# %%
L.seed_everything(42, workers=True)

# %%
# Load model architechture
from Models.AE_CNN import CNNAutoencoder, CNNAutoencoderWithLoss
from Models.LSTMClasifier import LSTMClassifier, LSTMClassifierWithLoss

# %%
# Load in AE weights for encoder
CHECKPOINT_cnn = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\CNN_AE_SpO2_FiO2\64_MSE"
)

model_AE = CNNAutoencoder.load_from_checkpoint(CHECKPOINT_cnn)

# Freeze encoder weights

for param in model_AE.parameters():
    param.requires_grad = False

# %%
model = LSTMClassifier(
    input_size=2,
    embedding_size=model_AE.latent_size,
    hidden_size=32,
    num_layers=1,
    Aencoder=model_AE,
    dropout_perc=0.25,
)

# %%
# Load in dataset
train_dataloader = DataLoader(dataset_train, batch_size=60, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=60)


# %%
# Settings for Lightning trainer
logger_path = Path(r"Z:/joris.vandervorst/logs")
model_name = "LSTM_Clasifier_SpO2_FiO2"
run_name = "CNN_64_hidden_32"
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
