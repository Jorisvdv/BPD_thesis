"""
Helper files for models predicting BPD

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-13

"""
# <codecell> Packages
# Import packages

import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (  # roc_curve,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # , cross_validate

# from sklearn import preprocessing

# # Print if import is a succes
# print("Import succes")
# <codecell> Settings
# Script settings

UTILITIES_SCRIPT_LOCATION = "Z:/joris.vandervorst/Scripts_Joris/Utilities.py"

DATA_FOLDER = "data"  # Specify folder name
STATIC_FILE_NAME = "testBPD3.csv"  # Specify file name

PROCESSED_DATA_FOLDER = "processed_data"  # Specify folder name
PROCESSED_STATIC_FILE_NAME = "testBPD3_cleaned.pkl"  # Specify file name

MODEL_FOLDER = "models"
METRICS_FOLDER = "metrics"

RANDOM_STATE = 42

SCORING_INNER = ["accuracy", "roc_auc", "f1"]
REFIT_INNER = "roc_auc"

# <codecell> Import data
# Import data


# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython
def load_static(file_location=None, cleaned_data=False):
    if file_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent

        if cleaned_data:
            file_location = (
                project_location / PROCESSED_DATA_FOLDER / PROCESSED_STATIC_FILE_NAME
            )
        else:
            file_location = project_location / DATA_FOLDER / STATIC_FILE_NAME

    # Change loading function based on suffix
    if file_location.suffix == ".csv":
        # Read csv file and return pandas file with data
        return pd.read_csv(file_location)

    if file_location.suffix == ".pkl":
        # Read pkl file and return pandas file with data
        with open(file_location, "rb") as f:
            return pickle.load(f)


# <codecell> Evaluation metrics
def calculateMetrics(inner, X_test, y_test):
    """
    Calculate score metrics and curves for a given GridSearchCV object and test data.

    Parameters:
    inner (sklearn.model_selection.GridSearchCV): The GridSearchCV object trained on the inner loop.
    X_test (pandas.DataFrame): The feature matrix for the test data.
    y_test (pandas.Series): The target variable for the test data.

    Returns:
    metrics (dict): A dictionary containing the score metrics and curves.
    """
    y_pred = inner.predict(X_test)
    y_pred_proba = inner.predict_proba(X_test)

    metrics = dict()
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
    metrics["f1"] = f1_score(y_test, y_pred)

    # Create roc curve using ROCCurveDisplay
    roc_curve_plot = RocCurveDisplay.from_estimator(inner, X_test, y_test)
    plt.close()  # Prevent plot from showing in notebook

    # Create calibration curve using CalibrationDisplay
    calibration_curve_plot = CalibrationDisplay.from_estimator(inner, X_test, y_test)
    plt.close()  # Prevent plot from showing in notebook

    metrics["roc_curve"] = roc_curve_plot
    metrics["calibration_curve"] = calibration_curve_plot

    return metrics


# <codecell> Nested CV and evaluate model


def nested_CV(model, X, y, parameters=None, inner_cv=5, outer_cv=5, verbose=0):
    """
    Perform nested cross-validation to evaluate a model's performance.

    Parameters:
    model (sklearn estimator): The model to evaluate.
    X (pandas.DataFrame): The feature matrix.
    y (pandas.Series): The target variable.
    parameters (dict, optional): The hyperparameters to tune using grid search.
    inner_cv (int, optional): The number of folds for the inner cross-validation loop.
    outer_cv (int, optional): The number of folds for the outer cross-validation loop.
    verbose (int, optional): The verbosity level of the output.

    Returns:
    results (dict): A dictionary containing the evaluation results. The dictionary has the following keys:
        - 'outer_scores' (numpy.ndarray): A 1D array of scores for each outer fold.
        - 'inner_scores' (numpy.ndarray): A 2D array of scores for each inner fold. The first dimension corresponds to the outer fold, and the second dimension corresponds to the inner fold.
        - 'best_params' (dict): The best hyperparameters found by grid search.
        - 'test_score' (float): The test score of the best model.
    """
    outer_scores = np.empty(outer_cv)
    inner_scores = np.empty((outer_cv, inner_cv))
    estimators = list()
    features = list()
    roc_curves = list()
    calibration_curves = list()

    #
    if parameters is None:
        parameters = dict()

    # Create inner and outer StratifiedKFold loops (to keep random state the save and in order to change outer loop in th future)
    inner_cv_folds = StratifiedKFold(
        n_splits=inner_cv, shuffle=True, random_state=RANDOM_STATE
    )
    outer_cv_folds = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=RANDOM_STATE
    )

    # Create pd.DataFrame to store models and scores
    model_scores = pd.DataFrame(
        {
            "test_accuracy": pd.Series(dtype="float"),
            "test_roc_auc": pd.Series(dtype="float"),
            "test_f1": pd.Series(dtype="float"),
            "estimator": pd.Series(dtype="object"),
            "features": pd.Series(dtype="object"),
            "roc_curve": pd.Series(dtype="object"),
            "calibration_curve": pd.Series(dtype="object"),
        },
        index=range(outer_cv),
    )

    # Loop over outer folds
    for fold, (trainidx, testidx) in enumerate(outer_cv_folds.split(X, y)):
        X_train, y_train = X.loc[X.index[trainidx]], y.loc[y.index[trainidx]]
        X_test, y_test = X.loc[X.index[testidx]], y.loc[y.index[testidx]]

        # Run inner loop using classifier (GridSearchCV object)
        classifier = GridSearchCV(
            model,
            parameters,
            cv=inner_cv,
            scoring=SCORING_INNER,
            refit=REFIT_INNER,
            verbose=verbose,
        )
        inner = classifier.fit(X_train, y_train)
        # Extract best estimator from inner loop trained on full dataset (refit=True)
        # Save estimator in model scores
        model_scores.at[fold, "estimator"] = inner.best_estimator_

        # Save features used in model in model scores
        model_scores.at[fold, "features"] = X_train.columns

        # Calculate score metrics
        metrics = calculateMetrics(inner, X_test, y_test)

        # Calculate score metrics and save
        model_scores.at[fold, "test_accuracy"] = metrics["accuracy"]
        model_scores.at[fold, "test_roc_auc"] = metrics["roc_auc"]
        model_scores.at[fold, "test_f1"] = metrics["f1"]
        # Save roc curve in model scores
        model_scores.at[fold, "roc_curve"] = metrics["roc_curve"]
        # Save calibration curve in model scores
        model_scores.at[fold, "calibration_curve"] = metrics["calibration_curve"]

    # Outer loop using scikit learn own cross_validate loop
    # cross_validate runs the outer loop and the classifier (GridSearchCV) runs the inner loop
    # Hardcoding scoring parameters
    # scoring = ["accuracy", "roc_auc", "f1"]
    # results = cross_validate(
    #     classifier, X, y, cv=outer_cv_folds, scoring=scoring, return_estimator=True
    # )
    # print(results)

    return model_scores


# <codecell> Print scores


def print_scores(model_scores):
    accuracy, auc_score, f1 = (
        model_scores["test_accuracy"],
        model_scores["test_roc_auc"],
        model_scores["test_f1"],
    )

    print(f"Accuracy score {accuracy.mean():%}, sd {accuracy.std():%}")
    print(f"AUC score: {auc_score.mean():.4f}, sd {auc_score.std():.4f}")
    print(f"F1 score: {f1.mean():.4f}, sd {f1.std():.4f}")


# <codecell> Plot roc curve
def create_roc_curve(model_scores, model_name=None):
    """
    Creates a ROC curve plot from the model scores dictionary.

    Parameters:
    model_scores (dict): A dictionary containing the model scores.
    model_name (str, optional): The name of the model to include in the plot title.

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object containing the ROC curve plot.
    """

    # Create a subplot to plot all roc curves
    fig, ax = plt.subplots()

    # Create numpy array of all tpr and values for use in calculating mean tpr and auc
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = np.empty((mean_fpr.shape[0], len(model_scores["roc_curve"])), dtype=float)
    all_auc = np.empty((len(model_scores["roc_curve"])), dtype=float)

    # Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
    for fold, roc_curve_plot in enumerate(model_scores["roc_curve"]):
        roc_curve_plot.plot(ax=ax, alpha=0.7)
        # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
        all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
        # Set first value of tpr to 0
        all_tpr[0, fold] = 0.0
        # Append auc to all_acu
        all_auc[fold] = roc_curve_plot.roc_auc

    # Plot random guessing line
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.7)

    # Plot mean ROC curve

    mean_tpr = np.mean(all_tpr, axis=1)
    mean_auc = np.mean(all_auc)
    std_auc = np.std(all_auc)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
        lw=2,
        alpha=0.9,
    )

    # Plot standard deviation area
    std_tpr = np.std(all_tpr, axis=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=f"\u00B1 1 SD",
    )

    # # Set axis limits
    # ax.set(
    #     xlim=[-0.05, 1.05],
    #     ylim=[-0.05, 1.05],
    # )

    # Add title and legend
    ax.set_title(f"ROC curve for {model_name}")
    ax.legend()  # loc="lower right")

    # TODO Add subtitle with selected parameters

    # Return figure
    return fig


# <codecell> Plot calibration curve
def create_calibration_curve(model_scores, model_name=None):
    """
    Creates a calibration curve plot from the model scores dictionary.

    Parameters:
    model_scores (dict): A dictionary containing the model scores.
    model_name (str, optional): The name of the model to include in the plot title.

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object containing the calibration curve plot.
    """

    # Create a subplot to plot all calibration curves
    fig, ax = plt.subplots()

    # Iterate over all calibration curves and plot on ax
    for fold, calibration_curve_plot in enumerate(model_scores["calibration_curve"]):
        calibration_curve_plot.plot(ax=ax, alpha=0.7)

    # Return figure
    # Add title and legend
    ax.set_title(f"Calibration curve for {model_name}")
    ax.legend()  # loc="lower right")

    # TODO Add subtitle with selected parameters

    return fig


# <codecell> Export model and metrics
# Export model and load model


def export_model_and_scores(
    name_model,
    folder_location=None,
    model=None,
    model_scores=None,
    selected_parameters=None,
):
    if folder_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        folder_location = script_location.parent.parent

    time_now = datetime.datetime.now()
    date_time = time_now.strftime("%Y%m%d_%H%M%S")

    model_folder_location = folder_location / MODEL_FOLDER
    metrics_folder_location = folder_location / METRICS_FOLDER

    if model is not None:
        # Export model to pickle file
        model_file_name = name_model + "_" + date_time + ".pkl"
        with open(model_folder_location / model_file_name, "wb") as f:
            pickle.dump(model, f)

    if model_scores is not None:
        # Export model to txt file
        metrics_file_name = name_model + "_" + date_time

        accuracy, auc_score, f1 = (
            model_scores["test_accuracy"],
            model_scores["test_roc_auc"],
            model_scores["test_f1"],
        )

        with open(
            metrics_folder_location / (metrics_file_name + ".txt"),
            "w",
            encoding="utf-8",
        ) as f:
            print(f"Model type: {name_model}", file=f)
            if selected_parameters is not None:
                print("Selected parameters:", *selected_parameters, file=f)
            print(f"AUC score: {auc_score.mean():4f}, sd {auc_score.std():4f}", file=f)
            print(f"Accuracy score {accuracy.mean():%}, sd {accuracy.std():%}", file=f)
            print(f"F1 score: {f1.mean():4f}, sd {f1.std():4f}", file=f)

            # print("Precision: ", precision.mean(), file = f)
            # print("Recall: ", recall.mean(), file = f)

        # Save model ROC curve plot
        create_roc_curve(model_scores, model_name=name_model).savefig(
            metrics_folder_location / (metrics_file_name + "_ROC" + ".png")
        )

        # Save calibration curve plot
        create_calibration_curve(model_scores, model_name=name_model).savefig(
            metrics_folder_location / (metrics_file_name + "_calibration" + ".png")
        )


# <codecell> Load model
def load_model(model_file):
    # Covert model file name to path
    model_file = Path(model_file)

    # Check if path is absolute,
    # otherwise use relative location defined in this script
    if not model_file.is_absolute():
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(UTILITIES_SCRIPT_LOCATION)
        )
        project_location = script_location.parent.parent

        # Choose most recent file of model name
        # In oder to pick specific time stamp specify exact name
        # otherwise the most recently created file that matches the regex is chosen
        model_folder_location = project_location / MODEL_FOLDER
        model_files = model_folder_location.glob(f"{model_file}*.pkl")
        model_file = max(model_files, key=lambda item: item.stat().st_ctime)

    # Load model from pickle file
    with open(model_file, "rb") as f:
        return pickle.load(f)
