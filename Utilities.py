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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (RocCurveDisplay, accuracy_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate)

# from sklearn import preprocessing

# # Print if import is a succes
# print("Import succes")
# <codecell> Settings
# Script settings

utilities_script_location = "Z:/joris.vandervorst/Scripts_Joris/Utilities.py"

data_folder = "data"  # Specify folder name
file_name = "testBPD3.csv"  # Specify file name

processed_data_folder = "processed_data"  # Specify folder name
processed_file_name = "testBPD3_cleaned.pkl"  # Specify file name

model_folder = "models"
metrics_folder = "metrics"

random_state = 42

# <codecell> Import data
# Import data

# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython
def load_static(file_location=None, cleaned_data=False):

    if file_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(utilities_script_location)
        )
        project_location = script_location.parent.parent

        if cleaned_data:
            file_location = (
                project_location / processed_data_folder / processed_file_name
            )
        else:
            file_location = project_location / data_folder / file_name

    # Change loading function based on suffix
    if file_location.suffix == ".csv":
        # Read csv file and return pandas file with data
        return pd.read_csv(file_location)

    if file_location.suffix == ".pkl":
        # Read pkl file and return pandas file with data
        with open(file_location, "rb") as f:
            return pickle.load(f)


# <codecell> Nested CV and evaluate model


def nested_CV(model, X, y, parameters=None, inner_cv=5, outer_cv=5, verbose=0):
    if parameters is None:
        parameters = dict()

    # Create inner and outer StratifiedKFold loops (to keep random state the save and in order to change outer loop in th future)
    inner_cv_folds = StratifiedKFold(
        n_splits=inner_cv, shuffle=True, random_state=random_state
    )
    outer_cv_folds = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=random_state
    )

    # Inner loop classifier
    classifier = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=inner_cv_folds,
        refit=True,
        n_jobs=(-1),
        verbose=verbose,
    )

    # Hardcoding scoring parameters for now
    scoring = ["accuracy", "roc_auc", "f1"]

    # TO DO: Adjust outer loop to StratifiedKFold
    # Keep output as dict with the following items:
    # model_scores["test_accuracy"]: list of scores for each fold scored by accuracy_score
    # model_scores["test_roc_auc"]: list of scores for each fold scored by roc_auc_score
    # model_scores["test_f1"]: list of scores for each fold scored by f1_score

    # Option: create empty np arrays and fill using fold-1 indexing (np.empty(inner_cv))

    # model_scores = defaultdict(list)

    # for fold, (trainidx, testidx) in enumerate(outer_cv_folds.split(X, y)):
    #     X_train, y_train = X.loc[X.index[trainidx]], y.loc[y.index[trainidx]]
    #     X_test, y_test = X.loc[X.index[testidx]], y.loc[y.index[testidx]]

    #     # Run inner loop using classifier (GridSearchCV object)
    #     inner = classifier.fit(X_train, y_train)
    #     # Extract best estimator from inner loop trained on full dataset (refit=True)
    #     best_inner = inner
    #     # Save estimator in model scores
    #     model_scores["estimator"].append(best_inner)

    #     # Predict and get predicted class and probability scores for use in scoring
    #     y_pred = inner.predict(X_test)
    #     y_pred_proba = inner.predict_proba(X_test)

    #     # Calculate score metrics and save in defaultdict
    #     model_scores["test_accuracy"].append(accuracy_score(y_test, y_pred))
    #     model_scores["test_roc_auc"].append(roc_auc_score(y_test, y_pred))
    #     model_scores["test_f1"].append(f1_score(y_test, y_pred))

    # print(model_scores)
    # TO DO: Create roc_curve output for combined roc curves

    # Outer loop using scikit learn own cross_validate loop
    # cross_validate runs the outer loop and the classifier (GridSearchCV) runs the inner loop
    # Hardcoding scoring parameters
    scoring = ["accuracy", "roc_auc", "f1"]
    results = cross_validate(
        classifier, X, y, cv=outer_cv_folds, scoring=scoring, return_estimator=True
    )
    # print(results)

    return results


def print_scores(model_scores):
    accuracy, auc_score, f1 = (
        model_scores["test_accuracy"],
        model_scores["test_roc_auc"],
        model_scores["test_f1"],
    )

    print(f"Accuracy score {accuracy.mean():%}, sd {accuracy.std():%}")
    print(f"AUC score: {auc_score.mean():4f}, sd {auc_score.std():4f}")
    print(f"F1 score: {f1.mean():4f}, sd {f1.std():4f}")


# TO DO: Plot ROC curve using roc_curve key from model_scores (not yet created)
# def plot_roc_curve(model_scores):
# plt.plot(fpr, tpr, linewidth=2, label=None)
# plt.plot([0, 1], [0, 1], "k--")
# plt.axis([0, 1, 0, 1])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.show()
# <codecell> Export model and metrics
# Export model and load model


def export_model_and_scores(
    name_model, folder_location=None, model=None, model_scores=None, selected_parameters=None
):

    if folder_location is None:
        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(utilities_script_location)
        )
    # Create string with date and time to append to file name
    time_now = datetime.datetime.now()
    date_time = time_now.strftime("%Y%m%d_%H%M%S")

    project_location = script_location.parent.parent
    model_folder_location = project_location / model_folder
    metrics_folder_location = project_location / metrics_folder

    if model is not None:
        # Export model to pickle file
        model_file_name = name_model + "_" + date_time + ".pkl"
        with open(model_folder_location / model_file_name, "wb") as f:
            pickle.dump(model, f)

    if model_scores is not None:
        # Export model to txt file
        metrics_file_name = name_model + "_" + date_time + ".txt"

        accuracy, auc_score, f1 = (
            model_scores["test_accuracy"],
            model_scores["test_roc_auc"],
            model_scores["test_f1"],
        )

        with open(metrics_folder_location / metrics_file_name, "w") as f:
            print(f"Model type: {name_model}", file=f)
            if selected_parameters is not None:
                print(f"Selected parameters:",*selected_parameters, file=f)
            print(f"AUC score: {auc_score.mean():4f}, sd {auc_score.std():4f}", file=f)
            print(f"Accuracy score {accuracy.mean():%}, sd {accuracy.std():%}", file=f)
            print(f"F1 score: {f1.mean():4f}, sd {f1.std():4f}", file=f)

            # print("Precision: ", precision.mean(), file = f)
            # print("Recall: ", recall.mean(), file = f)


def load_model(model_file):

    # Covert model file name to path
    model_file = Path(model_file)

    # Check if path is absolute, otherwise use relative location defined in this script
    if not model_file.is_absolute():

        script_location = (
            Path(__file__)
            if "__file__" in globals()
            else Path(utilities_script_location)
        )
        project_location = script_location.parent.parent

        # Choose most recent file of model name
        # In oder to pick specific time stamp specify exact name, otherwise the most recently created file that matches the regex is chosen
        model_folder_location = project_location / model_folder
        model_files = model_folder_location.glob(f"{model_file}*.pkl")
        model_file = max(model_files, key=lambda item: item.stat().st_ctime)

    # Load model from pickle file
    with open(model_file, "rb") as f:
        return pickle.load(f)


#     # Save plot
# Create string with date and time to append to file name
# time_now = datetime.datetime.now()
# date_time = time_now.strftime("%Y%m%d_%H%M%S")
#     plot_name = name_models + date_time + '.png'
#     plt.savefig(project_location / metrics_folder /plot_name)
