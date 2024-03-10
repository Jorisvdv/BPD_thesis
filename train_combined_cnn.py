"""
Training loop for Classifier model using a combination of static and temporal data

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

FREEZE_WEIGHTS = True

epochs = 1000


# %%
# Load model architechture
from Models.AE_CNN import CNNAutoencoder, CNNAutoencoderWithLoss
from Models.CombinedClasifier import CombinedClassifierCNN

# %%
# Define used model checkpoints
CHECKPOINT_encoder = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\CNN_AE_SpO2_FiO2\64_MSE"
)
static_checkpoint = getRecentCheckpoint(r"Z:\joris.vandervorst\logs\NN_Clasifier\64__4")
LSTM_checkpoint = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\LSTM_Clasifier_SpO2_FiO2\CNN_64_hidden_32_layers_1"
)


# Instantiate AE_model
model_AE = CNNAutoencoder.load_from_checkpoint(CHECKPOINT_encoder)

# Extract Hyperparameters for use in instatiation of Combined model
nn_params = torch.load(static_checkpoint)["hyper_parameters"]
lstm_params = torch.load(LSTM_checkpoint)["hyper_parameters"]

# %%
model = CombinedClassifierCNN(
    num_temoral_features=lstm_params["input_size"],
    num_static_featues=nn_params["input_size"],
    static_layer_size=nn_params["layer_1_size"],
    embedding_size=lstm_params["embedding_size"],
    hidden_size=lstm_params["hidden_size"],
    num_layers=lstm_params["num_layers"],
    AEncoder=model_AE,
    dropout_perc=0.25,
    lr=1e-3,
)

# %%
# Feed weights to model

# NN Model
# Get model weights from checkpoint
static_weights = torch.load(static_checkpoint)["state_dict"]
# Remove last weight and bias
remove_staic = ["fc3.weight", "fc3.bias"]
[static_weights.pop(key) for key in remove_staic]
# Rename for use in combined model
static_weights = {f"static_{key}": value for key, value in static_weights.items()}

# LSTM Clasifier

# # Get model weights from checkpoint
clasifier_weights = torch.load(LSTM_checkpoint)["state_dict"]
# Remove last weight
remove_clasifier = ["fc3.weight", "fc3.bias"]
[clasifier_weights.pop(key) for key in remove_staic]
# Rename for use in combined model
clasifier_fc = {
    f"temporal_{key}": value for key, value in clasifier_weights.items() if "fc2" in key
}
# Extract LSTM from clasifier
clasifier_lstm = {
    f"{key}": value for key, value in clasifier_weights.items() if "lstm" in key
}

# Create dict with model weights
combined_weights = static_weights | clasifier_fc | clasifier_lstm

# Feed weights to model (IncompatibleKeys error is expected)
model.load_state_dict(combined_weights, strict=False)

# %%
if FREEZE_WEIGHTS:
    # Freeze encoder weights
    for param in model.AEncoder.parameters():
        param.requires_grad = False
    # # Freeze Static weights
    for param in list(model.static_fc1.parameters()) + list(
        model.static_fc2.parameters()
    ):
        param.requires_grad = False
    # # Freeze LSTM_clasifier weights
    for param in list(model.lstm.parameters()) + list(model.temporal_fc2.parameters()):
        param.requires_grad = False
else:
    model.lr = 1e-4

# %%
# Load in dataset

train_dataloader = DataLoader(dataset_train, batch_size=60, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=60)

# %%
# Settings for Lightning trainer
logger_path = Path(r"Z:/joris.vandervorst/logs")
model_name = "Combined_Clasifier_SpO2_FiO2"
run_name = f"three_layers_NN64_test"
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
    max_epochs=epochs,
    logger=[TBLogger],
    log_every_n_steps=5,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,  # Uncomment for quick loop of 1 epoch without logs
    max_time={"hours": 24, "minutes": 0},
    # accumulate_grad_batches=30,
    gradient_clip_val=1,
)

# %%
trainer.fit(model, train_dataloader, val_dataloader)


# %%
#  Rerun model with unfrozen layer for LSTM

CHECKPOINT_model = getRecentCheckpoint(
    "Z:/joris.vandervorst/logs/" + model_name + "/" + run_name
)
model.lr = 1e-6
# Unfreeze LSTM weights
for param in list(model.lstm.parameters()) + list(model.temporal_fc2.parameters()):
    param.requires_grad = True

TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name=model_name,
    version=run_name + "_unfrozen_lstm",
    log_graph=True,
)
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)

# %%
trainer = L.Trainer(
    max_epochs=epochs,
    logger=[TBLogger],
    log_every_n_steps=5,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,
    max_time={"hours": 10, "minutes": 0},
    # default_root_dir=logger_path,
    # accumulate_grad_batches=30,
    gradient_clip_val=1,
)
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=CHECKPOINT_model)


# %%
#  Rerun model with unfrozen layer for LSTM and static layer

CHECKPOINT_model = getRecentCheckpoint(
    "Z:/joris.vandervorst/logs/" + model_name + "/" + run_name + "_unfrozen_lstm"
)
model.lr = 1e-6
# Unfreeze  Static weights
for param in list(model.static_fc1.parameters()) + list(model.static_fc2.parameters()):
    param.requires_grad = True

TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name=model_name,
    version=run_name + "_unfrozen_lstm" + "_unfrozen_static",
    log_graph=True,
)
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)

# %%
trainer = L.Trainer(
    max_epochs=epochs,
    logger=[TBLogger],
    log_every_n_steps=5,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,
    max_time={"hours": 10, "minutes": 0},
    # default_root_dir=logger_path,
    # accumulate_grad_batches=30,
    gradient_clip_val=1,
)
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=CHECKPOINT_model)

# %%


#  Rerun model with all weights unfrozen

CHECKPOINT_model = getRecentCheckpoint(
    "Z:/joris.vandervorst/logs/"
    + model_name
    + "/"
    + run_name
    + "_unfrozen_lstm"
    + "_unfrozen_static"
)
model.lr = 1e-8
# Unfreeze all weights
for param in list(model.parameters()):
    param.requires_grad = True

TBLogger = TensorBoardLogger(
    save_dir=logger_path,
    name=model_name,
    version=run_name + "_unfrozen_all",
    # sub_dir=,
    log_graph=True,
)
checkpoint = ModelCheckpoint(monitor="loss/val", mode="min")
earlystopping = EarlyStopping(monitor="loss/val", mode="min", patience=50)
# %%
trainer = L.Trainer(
    max_epochs=epochs,
    logger=[TBLogger],
    log_every_n_steps=5,
    callbacks=[checkpoint, earlystopping],
    # fast_dev_run=True,
    max_time={"hours": 1, "minutes": 0},
    gradient_clip_val=1,
)
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=CHECKPOINT_model)

# %%
