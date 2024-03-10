"""
Script to calculate metrics on test set

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-09-19

"""

# %%
import dataclasses
import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (  # roc_curve,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from cached_data import dataset_test, dataset_test_SpO2, dataset_train, df_test
from Dataloaders.CombinedDataLoader import PatientDatasetMem, input, target
from Models.AE_LSTM_double import RecurrentAutoencoderWithLoss
from Models.CNNAutoencoder import CNNAutoencoder

# %%
# Load model architechtures
from Models.CombinedClasifier import CombinedClassifierCNN, CombinedClassifierWithLoss
from Models.LSTMClasifier import LSTMClassifier
from Models.NNClasifier import BinaryClassifier
from Utils.metrics import Metrics

results_folder = Path("Z:/joris.vandervorst/reports/Results/NN")


# %%
def save_metrics(metrics_list, filename, folder=results_folder):
    # output as dict
    output_dict = {
        metrics.name: dataclasses.asdict(metrics) for metrics in metrics_list
    }
    # Convert to json
    json_object = json.dumps(output_dict, indent=4)

    json_location = Path(folder) / f"metrics_{filename}.json"

    with open(json_location, "w") as jsonfile:
        jsonfile.write(json_object)


# %%
def getRecentCheckpoint(folder):
    location = Path(folder)
    checkpoints = list(location.rglob("*.ckpt"))
    checkpoints.sort(key=lambda file: file.stat().st_mtime)
    # Get most recent checkpoint and parameters
    return checkpoints[-1]


# %%
def calculate_metric(y_true, y_pred_proba, model_name="", dataset_name=""):
    y_pred = y_pred_proba.round()
    metrics = Metrics(
        name=f"{model_name}_{dataset_name}",
        roc_auc=float(roc_auc_score(y_test, y_pred_proba)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, pos_label=1, average="binary")),
        true_values=y_test.tolist(),
        predicted_probabilities=y_pred_proba.tolist(),
    )
    return metrics


# %%
# Select checkpoint of models that had the best performance on the validation set
checkpoint_static = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\NN_Clasifier\64_auc"
)
checkpoint_AE = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\CNN_AE_SpO2_FiO2\64_MSE"
)
checkpoint_LSTM = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\LSTM_Clasifier_SpO2_FiO2\CNN_64_hidden_32_layers_1"
)
checkpoint_Combined = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\Combined_Clasifier_SpO2_FiO2\three_layers"
)
checkpoint_Combined_fix = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\Combined_Clasifier_SpO2_FiO2\three_layers_NN64_fix"
)
checkpoint_Combined_fix_lstm = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\Combined_Clasifier_SpO2_FiO2\three_layers_NN64_fix_unfrozen_lstm"
)
checkpoint_Combined_fix_static = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\Combined_Clasifier_SpO2_FiO2\three_layers_NN64_fix_unfrozen_lstm_unfrozen_static"
)
checkpoint_Combined_fix_all = getRecentCheckpoint(
    r"Z:\joris.vandervorst\logs\Combined_Clasifier_SpO2_FiO2\three_layers_NN64_fix_unfrozen_all"
)

# %%
# Instantiate models
model_cnn_AE = CNNAutoencoder.load_from_checkpoint(checkpoint_AE).eval()
model_cnn_AE = CNNAutoencoder.load_from_checkpoint(checkpoint_AE).eval()
model_static = BinaryClassifier.load_from_checkpoint(checkpoint_static).eval()
model_lstm = LSTMClassifier.load_from_checkpoint(
    checkpoint_LSTM, Aencoder=model_cnn_AE
).eval()
# model_combined = CombinedClassifierCNN.load_from_checkpoint(
#     checkpoint_Combined, AEncoder=model_cnn_AE
# ).eval()
model_combined = CombinedClassifierCNN.load_from_checkpoint(
    checkpoint_Combined_fix_lstm, AEncoder=model_cnn_AE
).eval()
# model_combined_fix = CombinedClassifierCNN.load_from_checkpoint(
#     checkpoint_Combined_fix, AEncoder=model_cnn_AE
# ).eval()
# model_combined_fix_lstm = CombinedClassifierCNN.load_from_checkpoint(
#     checkpoint_Combined_fix_lstm, AEncoder=model_cnn_AE
# ).eval()
# model_combined_fix_static = CombinedClassifierCNN.load_from_checkpoint(
#     checkpoint_Combined_fix_static, AEncoder=model_cnn_AE
# ).eval()
# model_combined_fix_all = CombinedClassifierCNN.load_from_checkpoint(
#     checkpoint_Combined_fix_all, AEncoder=model_cnn_AE
# ).eval()

# AutoEncoder_model = RecurrentAutoencoderWithLoss.load_from_checkpoint(
#     CHECKPOINT_encoder, n_features=1, embedding_dim=16, seq_len=120
# )
# model_combined_loss = CombinedClassifierWithLoss.load_from_checkpoint(
#     checkpoint_Combined_loss, AEncoder=AutoEncoder_model
# ).eval()

# %%
Metics_dict = {}

# %%
# x, seq, y_test = next(iter(DataLoader(dataset_test, batch_size=500, shuffle=False)))


# dataset_name = "full"
# with torch.no_grad():
#     metrics_static = calculate_metric(
#         y_true=y_test, y_pred_proba=model_static.forward(x), model_name="NN_Static", dataset_name=dataset_name,
#     )
#     metrics_lstm = calculate_metric(y_true=y_test, y_pred_proba=model_lstm.forward(seq), model_name="NN_LSTM", dataset_name=dataset_name,)
#     metrics_combined = calculate_metric(
#         y_true=y_test, y_pred_proba=model_combined.forward(x, seq), model_name="NN_Combined", dataset_name=dataset_name,
#     )
# metrics_list=[metrics_static, metrics_lstm, metrics_combined]
# print(f"Dataset: {dataset_name}")
# for metric in metrics_list:
#     print(f"Model: {metric.name}\t AUC: {metric.roc_auc:4f}\t Accuracy: {metric.accuracy:%}\t F1: {metric.f1:4f}")

# Metics_dict[dataset_name] = metrics_list
# save_metrics(metrics_list=metrics_list,filename=dataset_name)

# %%

# x_spo2, seq_spo2, y_test_spo2 = next(
#     iter(DataLoader(dataset_test_SpO2, batch_size=500, shuffle=False))
# )

# with torch.no_grad():
#     metrics_combined_loss = calculate_metric(
#         y_true=y_test_spo2, y_pred_proba=model_combined_loss.forward(x, seq_spo2)
#     )


# print(metrics_combined_loss)


# %%
for day in range(1, 8):
    dataset_day = PatientDatasetMem(
        dataframe=df_test,
        input_labels=input,
        target_label=target,
        preprocessor=dataset_train.preprocessor,
        daysToLoad=day,
    )
    dataset_day.temp_mean = dataset_test.temp_mean
    dataset_day.temp_std = dataset_test.temp_std
    dataset_day.scale_temp()
    x, seq, y_test = next(iter(DataLoader(dataset_day, batch_size=500)))

    dataset_name = f"Day_{day}"
    with torch.no_grad():
        metrics_static = calculate_metric(
            y_true=y_test,
            y_pred_proba=model_static.forward(x),
            model_name="NN_Static",
            dataset_name=dataset_name,
        )
        metrics_lstm = calculate_metric(
            y_true=y_test,
            y_pred_proba=model_lstm.forward(seq),
            model_name="NN_LSTM",
            dataset_name=dataset_name,
        )
        metrics_combined = calculate_metric(
            y_true=y_test,
            y_pred_proba=model_combined.forward(x, seq),
            model_name="NN_Combined",
            dataset_name=dataset_name,
        )
        # metrics_combined_fix = calculate_metric(
        #     y_true=y_test,
        #     y_pred_proba=model_combined_fix.forward(x, seq),
        #     model_name="NN_Combined_fix",
        #     dataset_name=dataset_name,
        # )
        # metrics_combined_fix_lstm = calculate_metric(
        #     y_true=y_test,
        #     y_pred_proba=model_combined_fix_lstm.forward(x, seq),
        #     model_name="NN_Combined_fix_lstm",
        #     dataset_name=dataset_name,
        # )
        # metrics_combined_fix_static = calculate_metric(
        #     y_true=y_test,
        #     y_pred_proba=model_combined_fix_static.forward(x, seq),
        #     model_name="NN_Combined_fix_static",
        #     dataset_name=dataset_name,
        # )
        # metrics_combined_fix_all = calculate_metric(
        #     y_true=y_test,
        #     y_pred_proba=model_combined_fix_all.forward(x, seq),
        #     model_name="NN_Combined_fix_all",
        #     dataset_name=dataset_name,
        # )
    metrics_list = [
        metrics_static,
        metrics_lstm,
        metrics_combined,
        # metrics_combined_fix,
        # metrics_combined_fix_lstm,
        # metrics_combined_fix_static,
        # metrics_combined_fix_all,
    ]
    print(f"Dataset: {dataset_name}")
    for metric in metrics_list:
        print(
            f"Model: {metric.name}\t AUC: {metric.roc_auc:4f}\t Accuracy: {metric.accuracy:%}\t F1: {metric.f1:4f}"
        )

    Metics_dict[dataset_name] = metrics_list

    save_metrics(metrics_list=metrics_list, filename=dataset_name)

# %%
total_dict = {}
for key, value in Metics_dict.items():
    total_dict[key] = {metrics.name: dataclasses.asdict(metrics) for metrics in value}
# %%
filename = "total"
# Convert to json
json_object = json.dumps(total_dict, indent=4)

json_location = Path(results_folder) / f"metrics_{filename}.json"

with open(json_location, "w") as jsonfile:
    jsonfile.write(json_object)
# %%
# with open(json_location, "r") as jsonfile:
#     test_json = json.load(jsonfile)
# print(np.array(test_json["Day_7"]['NN_Static_Day_7']['true_values']).shape)
# %%
