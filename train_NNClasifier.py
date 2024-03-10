# %%
from pathlib import Path

import lightning as L
import torch
from Dataloaders.CombinedDataLoader import (  # dataset_train_SpO2,; dataset_val_SpO2,
    PatientDatasetMem,
    input,
    target,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader
from Utils.cached_data import (
    dataset_train,
    dataset_train_SpO2,
    dataset_val,
    dataset_val_SpO2,
    df_test,
    df_train,
    df_val,
)

# %%
L.seed_everything(4, workers=True)
logger_path = Path(r"Z:/joris.vandervorst/logs")
# %%
from Models.NNClasifier import BinaryClassifier

# %%
model = BinaryClassifier(7, 32)
checkpoint = ModelCheckpoint(monitor="auc/val", mode="max")
earlystopping = EarlyStopping(monitor="auc/val", mode="max", patience=200)

# latest_checkpoint = list(
# (logger_path / r"lightning_logs\version_6\checkpoints").rglob("*.ckpt")
TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name="NN_Clasifier",
    version="32_auc",
    # sub_dir=,
    log_graph=True,
)
model.lr = 1e-3

# %%
# Load in dataset
train_dataloader = DataLoader(dataset_train, batch_size=60, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=60)

# %%
# Create a Trainer and train the model
trainer = L.Trainer(
    max_epochs=1000,
    logger=[TBLogger],
    log_every_n_steps=1,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,
    max_time={"hours": 5},
    default_root_dir=logger_path,
    # accumulate_grad_batches=64,
)
# %%
# Create tuner for finding learning rate
# tuner = Tuner(trainer)

# tuner.lr_find(model, train_dataloader)
# %%
trainer.fit(model, train_dataloader, val_dataloader)


# %%
