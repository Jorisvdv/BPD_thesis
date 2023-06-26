"""
Logistic regression of static dataset for the prediction of BPD

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-13

"""
# <codecell> Packages
# Import packages

import datetime

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from Utilities import (
    export_model_and_scores,
    load_model,
    load_static,
    nested_CV,
    print_scores,
)

# <codecell> Settings
# Script settings
name_model = "log_reg_static"
random_state = 42
export = True

# <codecell> Import data
# Import data

# Import data from csv file using pathlib to specify path relative to script
# find location of script and set manual location for use in ipython

# Read csv file
data_static = load_static(cleaned_data=True)

# <codecell> Data features
# Show column names
print(f" Column names: ")
print(*data_static.columns)

# columns_to_drop = ["Furosemide", "DEXA"]
# data_static = data_static.drop(columns_to_drop, axis=1)

# <codecell> Fit Logistic regression
# Logistic regression
selected_parameters = [
    "Furosemide",
    "DEXA",
]

# Split data into features and target
X = data_static.loc[:, selected_parameters]  # .drop("y", axis=1)
y = data_static["y"]

# Create logistic regression model
log_reg = LogisticRegression(multi_class="ovr", max_iter=10000)

# Define parameter search space
log_reg_parameters = {}  # {'penalty': ['l1', 'l2', 'elasticnet']}

model_scores = nested_CV(
    model=log_reg,
    parameters=log_reg_parameters,
    inner_cv=5,
    outer_cv=5,
    X=X,
    y=y,
    verbose=0,
)

# <codecell> Print Scores
# Print Scores

print_scores(model_scores)


# # Plot ROC curve

# plt.plot(fpr, tpr, linewidth=2, label=None)
# plt.plot([0, 1], [0, 1], "k--")
# plt.axis([0, 1, 0, 1])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.show()


# <codecell> Export model and metrics
# Export model

if export:
    model = model_scores["estimator"][
        model_scores["test_accuracy"].argmax()
    ].best_estimator_

    export_model_and_scores(
        name_model=name_model,
        model=model,
        model_scores=model_scores,
        selected_parameters=selected_parameters,
    )

    # # Save plot
    # plot_name = name_models + date_time + ".png"
    # plt.savefig(project_location / metrics_folder / plot_name)

# %%
