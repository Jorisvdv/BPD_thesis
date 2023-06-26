"""
Logistic regression of static dataset for the prediction of BPD

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-13

"""
## Import packages

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, cross_val_score

## Import data
# Import data from csv file using pathlib to specify path relative to script

# Manually specify path to data file if __file__ does not work


# find location of script and go up two levels to find main project directory
script_location = Path(__file__).absolute().parent.parent

# script_location = Path(__file__).absolute().parent
# # Find main project directory by going up one level
# project_location = script_location.parent
# data_folder = "data"  # Specify folder name
# file_name = "static_data.csv"  # Specify file name

# Read csv file
data_static = pd.read_csv(script_location / data_folder / file_name)

# Count number of rows and columns
data_static.shape

# Show first 5 rows
data_static.head()

# Show data types of columns
data_static.dtypes

# Check for missing values
data_static.isnull().sum()

# Check for duplicate rows
data_static.duplicated().sum()


## Data preprocessing
# Drop columns that are not needed for the model
columns_to_drop = ["id", "date", "time", "event", "BPD"]
data_static = data_static.drop(columns_to_drop, axis=1)


## Logistic regression
# Split data into features and target
X = data_static.drop("BPD_binary", axis=1)
y = data_static["BPD_binary"]

# Create logistic regression model
log_reg = LogisticRegression()

# Preprocess data for use in logistic regression
X = preprocessing.scale(X)

# Fit model using nested cross-validation
y_pred = cross_val_predict(log_reg, X, y, cv=5)

# Calculate AUC score
auc_score = roc_auc_score(y, y_pred)
print("AUC score: ", auc_score)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred)

# Calculate accuracy
accuracy = cross_val_score(log_reg, X, y, cv=5, scoring="accuracy")
print("Accuracy: ", accuracy.mean())

# Calculate precision
precision = cross_val_score(log_reg, X, y, cv=5, scoring="precision")
print("Precision: ", precision.mean())

# Calculate recall
recall = cross_val_score(log_reg, X, y, cv=5, scoring="recall")
print("Recall: ", recall.mean())

# Calculate F1 score
f1 = cross_val_score(log_reg, X, y, cv=5, scoring="f1")
print("F1 score: ", f1.mean())

# Plot ROC curve

plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot([0, 1], [0, 1], "k--")
plt.axis([0, 1, 0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


## Export model
# Export model to pickle file using pathlib to specify path relative to script
model_folder = "models"  # Specify folder name
model_name = "log_reg_static.pkl"  # Specify file name
pickle.dump(
    log_reg, open(script_location / model_folder / model_name, "wb")
)  # Export model

## Save model metrics to txt file
metrics_folder = "metrics"  # Specify folder name
# Create sring with date and time to append to file name
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
metrics_name = "log_reg_static_" + date_time + ".txt"  # Specify file name
# Create txt file
with open(script_location / metrics_folder / metrics_name, "w") as f:
    print("AUC score: ", auc_score, file=f)
    print("Accuracy: ", accuracy.mean(), file=f)
    print("Precision: ", precision.mean(), file=f)
    print("Recall: ", recall.mean(), file=f)
    print("F1 score: ", f1.mean(), file=f)

# Save plot
plot_name = "log_reg_static" + date_time + ".png"  # Specify file name
plt.savefig(script_location / metrics_folder / plot_name)
