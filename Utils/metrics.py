"""
Helper files for the calculation of metrics.

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-09-19

"""
# <codecell> Packages
# Import packages

import dataclasses
import json
import logging
import pickle
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from Models.modelclass import Model
from pandas import DataFrame, Series
from sklearn.metrics import (  # roc_curve,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import Tensor


# <codecell> Metrics
# Define metrics class
@dataclasses.dataclass
class Metrics:
    """
    Class for the metrics of a model.

    Parameters:
    accuracy (float): The accuracy score of the model.
    roc_auc (float): The roc_auc score of the model.
    f1 (float): The f1 score of the model.
    true_values (list[int]): The true values of the model.
    predicted_probabilities (list[float]): The predicted probabilities of the model.
    name (str, optional): The name of the model.
    """

    accuracy: float
    roc_auc: float
    f1: float
    true_values: list[int]
    predicted_probabilities: list[float]
    name: str = ""


# <codecell> Evaluation metrics
# Define function to calculate metrics
def calculateMetrics_from_model(
    model: Model,
    X_test: Union[DataFrame, np.ndarray, Tensor],
    y_test: Union[DataFrame, Series, np.ndarray, Tensor],
    model_name: str = "",
    dataset_name: str = "",
    specs: str = "",
) -> Metrics:
    """
    Calculate score metrics and curves for a given GridSearchCV object and test data.

    Parameters:
    model (modelclass.Model): The model object to use for prediction.
    X_test (Union[DataFrame, np.ndarray, Tensor]): The feature matrix for the test data.
    y_test (Union[DataFrame, np.ndarray, Tensor]): The target variable for the test data.

    Returns:
    metrics (Metrics): The metrics of the model.
    """

    y_pred_proba = model.predict_proba(X_test)

    # Reduce y_pred_proba to only the positive class
    if y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, -1]

    # Round y_pred_proba to get y_pred
    y_pred = y_pred_proba.round()

    # # Create roc curve using ROCCurveDisplay
    # # Create legend for plot
    # if legend is not None:
    #     legend = f"{legend}"
    # else:
    #     legend = None

    # roc_curve_plot = RocCurveDisplay.from_predictions(
    #     y_true=y_test, y_pred=y_pred_proba, name=legend
    # )
    # plt.close()  # Prevent plot from showing in notebook

    # Create metrics object
    metrics = Metrics(
        name=f"{model_name}_{dataset_name}_{specs}",
        roc_auc=float(roc_auc_score(y_test, y_pred_proba)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, pos_label=1, average="binary")),
        true_values=y_test.tolist(),
        predicted_probabilities=y_pred_proba.tolist(),
    )
    return metrics


# <codecell> plot and calculate metrics
def plot_and_calculate_metrics(
    outputs: list[Metrics], folder: Union[Path, str], model_name: str, dataset_name: str
):
    # Save output as pickle file
    # file_location = METRICS_FOLDER / f"{cfg.model.name_model}_{cfg.dataset_name}.pkl"
    file_location = Path(folder) / "metrics.pkl"

    with open(file_location, "wb") as file:
        pickle.dump(outputs, file)

    # output as dict
    output_dict_list = [dataclasses.asdict(metrics) for metrics in outputs]
    # Convert to json
    json_object = json.dumps(output_dict_list, indent=4)

    json_location = Path(folder) / "metrics.json"

    with open(json_location, "w") as jsonfile:
        jsonfile.write(json_object)

    # Collect metrics
    auc_score = []
    accuracy = []
    f1_score = []
    for result in outputs:
        auc_score.append(result.roc_auc)
        accuracy.append(result.accuracy)
        f1_score.append(result.f1)
    # Transform to numpy arrays
    auc_score = np.array(auc_score)
    accuracy = np.array(accuracy)
    f1_score = np.array(f1_score)
    # Calculate mean and standard deviation and log
    logging.info(f"AUC score: {auc_score.mean():.4f}, sd {auc_score.std():.4f}")
    logging.info(f"Accuracy score {accuracy.mean():%}, sd {accuracy.std():%}")
    logging.info(f"F1 score: {f1_score.mean():.4f}, sd {f1_score.std():.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, result in enumerate(outputs):
        if len(outputs) > 1:
            foldname = f"Fold {fold}"
        else:
            foldname = None
        RocCurveDisplay.from_predictions(
            y_true=result.true_values,
            y_pred=result.predicted_probabilities,
            ax=ax,
            name=foldname,
        )

        # .roc_curve.plot(
        #     ax=ax,
        #     # label=result[0].roc_auc,
        #     plot_chance_level=False,
        # )
    plt.legend()
    plt.title(f"ROC curves {model_name} {dataset_name}")
    plt.savefig(Path(folder) / "plot.png")
    plt.close()
