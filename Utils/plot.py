# %%
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from Utils.metrics import Metrics

# %%
with open(
    r"Z:\joris.vandervorst\outputs\2023-09-23\22-42-30\Static_no_meds_Logistic regression_feat_sel_True_scale_True\metrics.pkl",
    "rb",
) as file:
    LR_Static = pickle.load(file)
with open(
    r"Z:\joris.vandervorst\outputs\2023-09-23\22-42-30\Static_no_meds_Random Forest_feat_sel_True_scale_False\metrics.pkl",
    "rb",
) as file:
    RF_Static = pickle.load(file)
with open(
    r"Z:\joris.vandervorst\outputs\2023-09-24\20-45-45\Temporal_features_pca_7_Logistic regression_feat_sel_True_scale_False\metrics.pkl",
    "rb",
) as file:
    LR_7 = pickle.load(file)
with open(
    r"Z:\joris.vandervorst\outputs\2023-09-23\22-42-30\Temporal_features_day_7_Random Forest_feat_sel_True_scale_False\metrics.pkl",
    "rb",
) as file:
    RF_7 = pickle.load(file)
with open(
    r"Z:\joris.vandervorst\reports\Results\NN\metrics_Day_7.json",
    "r",
) as file:
    NN = json.load(file)


# %%
fig, ax = plt.subplots(figsize=(7, 7))


# Create numpy array of all tpr and values for use in calculating mean tpr and auc

mean_fpr = np.linspace(0, 1, 100)
all_tpr = np.empty((mean_fpr.shape[0], len(LR_Static)), dtype=float)
all_auc = np.empty((len(LR_Static)), dtype=float)

# Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
for fold, metrics in enumerate(LR_Static):
    roc_curve_plot = RocCurveDisplay.from_predictions(
        y_true=metrics.true_values,
        y_pred=metrics.predicted_probabilities,
        name=f"Logistic regression fold {fold+1}",
        ax=None,
    )
    # roc_curve_plot.plot(ax=ax, alpha=0.7)
    # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
    all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
    # Set first value of tpr to 0
    all_tpr[0, fold] = 0.0
    # Append auc to all_acu
    all_auc[fold] = roc_curve_plot.roc_auc

# Plot mean ROC curve

mean_tpr = np.mean(all_tpr, axis=1)
mean_auc = np.mean(all_auc)
std_auc = np.std(
    all_auc, ddof=1
)  # Specify using n-1 df in oder to use same std as pandas (pandas uses sample mean)

ax.plot(
    mean_fpr,
    mean_tpr,
    color="purple",
    label=f"Logistic regression static (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
    lw=1,
    alpha=0.9,
)

# # Create numpy array of all tpr and values for use in calculating mean tpr and auc
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = np.empty((mean_fpr.shape[0], len(RF_Static)), dtype=float)
# all_auc = np.empty((len(RF_Static)), dtype=float)

# # Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
# for fold, metrics in enumerate(RF_Static):
#     roc_curve_plot = RocCurveDisplay.from_predictions(
#         y_true=metrics.true_values,
#         y_pred=metrics.predicted_probabilities,
#         name=f"Logistic regression fold {fold+1}",
#         ax=None,
#     )
#     # roc_curve_plot.plot(ax=ax, alpha=0.7)
#     # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
#     all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
#     # Set first value of tpr to 0
#     all_tpr[0, fold] = 0.0
#     # Append auc to all_acu
#     all_auc[fold] = roc_curve_plot.roc_auc


# # Plot mean ROC curve

# mean_tpr = np.mean(all_tpr, axis=1)
# mean_auc = np.mean(all_auc)
# std_auc = np.std(
#     all_auc, ddof=1
# )  # Specify using n-1 df in oder to use same std as pandas (pandas uses sample mean)

# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=f"Random Forest (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
#     lw=2,
#     alpha=0.9,
# )
# # Create numpy array of all tpr and values for use in calculating mean tpr and auc
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = np.empty((mean_fpr.shape[0], len(LR_7)), dtype=float)
# all_auc = np.empty((len(LR_7)), dtype=float)

# # Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
# for fold, metrics in enumerate(RF_Static):
#     roc_curve_plot = RocCurveDisplay.from_predictions(
#         y_true=metrics.true_values,
#         y_pred=metrics.predicted_probabilities,
#         name=f"Logistic regression fold {fold+1}",
#         ax=None,
#     )
#     # roc_curve_plot.plot(ax=ax, alpha=0.7)
#     # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
#     all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
#     # Set first value of tpr to 0
#     all_tpr[0, fold] = 0.0
#     # Append auc to all_acu
#     all_auc[fold] = roc_curve_plot.roc_auc

# Create numpy array of all tpr and values for use in calculating mean tpr and auc
mean_fpr = np.linspace(0, 1, 100)
all_tpr = np.empty((mean_fpr.shape[0], len(LR_7)), dtype=float)
all_auc = np.empty((len(LR_7)), dtype=float)

# Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
for fold, metrics in enumerate(LR_7):
    roc_curve_plot = RocCurveDisplay.from_predictions(
        y_true=metrics.true_values,
        y_pred=metrics.predicted_probabilities,
        name=f"Logistic regression fold {fold+1}",
        ax=None,
    )
    # roc_curve_plot.plot(ax=ax, alpha=0.7)
    # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
    all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
    # Set first value of tpr to 0
    all_tpr[0, fold] = 0.0
    # Append auc to all_acu
    all_auc[fold] = roc_curve_plot.roc_auc

# Plot mean ROC curve

mean_tpr = np.mean(all_tpr, axis=1)
mean_auc = np.mean(all_auc)
std_auc = np.std(
    all_auc, ddof=1
)  # Specify using n-1 df in oder to use same std as pandas (pandas uses sample mean)

ax.plot(
    mean_fpr,
    mean_tpr,
    color="red",
    label=f"Logistic regression day 7 (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
    lw=1,
    alpha=0.9,
)
# # Create numpy array of all tpr and values for use in calculating mean tpr and auc
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = np.empty((mean_fpr.shape[0], len(RF_7)), dtype=float)
# all_auc = np.empty((len(RF_7)), dtype=float)

# # Iterate over all roc curves and plot on ax and save tpr values in mean_tpr
# for fold, metrics in enumerate(RF_7):
#     roc_curve_plot = RocCurveDisplay.from_predictions(
#         y_true=metrics.true_values,
#         y_pred=metrics.predicted_probabilities,
#         name=f"Logistic regression fold {fold+1}",
#         ax=None,
#     )
#     # roc_curve_plot.plot(ax=ax, alpha=0.7)
#     # Interpolate tpr values to mean_fpr because not all roc curves have the same amount of tpr values
#     all_tpr[:, fold] = np.interp(mean_fpr, roc_curve_plot.fpr, roc_curve_plot.tpr)
#     # Set first value of tpr to 0
#     all_tpr[0, fold] = 0.0
#     # Append auc to all_acu
#     all_auc[fold] = roc_curve_plot.roc_auc

# # Plot mean ROC curve

# mean_tpr = np.mean(all_tpr, axis=1)
# mean_auc = np.mean(all_auc)
# std_auc = np.std(
#     all_auc, ddof=1
# )  # Specify using n-1 df in oder to use same std as pandas (pandas uses sample mean)

# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=f"Random Forest day 7 (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
#     lw=2,
#     alpha=0.9,
# )

# # Plot standard deviation area

# std_tpr = np.std(all_tpr, axis=1)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=f"\u00B1 1 SD",
# )

# # Set axis limits
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)
roc_curve_plot = RocCurveDisplay.from_predictions(
    y_true=NN["NN_Static_Day_7"]["true_values"],
    y_pred=NN["NN_Static_Day_7"]["predicted_probabilities"],
    name=f"NN static data",
    ax=ax,
    color="cyan",
)
roc_curve_plot = RocCurveDisplay.from_predictions(
    y_true=NN["NN_LSTM_Day_7"]["true_values"],
    y_pred=NN["NN_LSTM_Day_7"]["predicted_probabilities"],
    name=f"LSTM-AE",
    ax=ax,
    color="green",
)
roc_curve_plot = RocCurveDisplay.from_predictions(
    y_true=NN["NN_Combined_Day_7"]["true_values"],
    y_pred=NN["NN_Combined_Day_7"]["predicted_probabilities"],
    name=f"Combined",
    ax=ax,
    color="orange",
)

# Plot random guessing line
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="grey", label="Chance", alpha=0.7)
ax.set_title(f"ROC Scores at day 7")
ax.legend()  # loc="lower right")
# plt.savefig("ROC_combined_7.svg")
# plt.savefig("ROC_combined_7.png")
plt.show()


# %%
def create_roc_curve(model_metrics_list, plot_name=None, ax=None):
    """
    Creates a ROC curve plot from the model metrics object.

    fig (matplotlib.figure.Figure): The matplotlib figure object containing the ROC curve plot.
    """

    # # Create a subplot to plot all roc curves
    # fig, ax = plt.subplots()

    # Create numpy array of all tpr and values for use in calculating mean tpr and auc
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = np.empty((mean_fpr.shape[0], len(model_metrics_list)), dtype=float)
    all_auc = np.empty((len(model_metrics_list)), dtype=float)

    roc_curve_plot = RocCurveDisplay.from_predictions(
        y_true=y_test, y_pred=y_pred_proba, name=plot_name, ax=ax
    )

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

    # # Plot mean ROC curve

    # mean_tpr = np.mean(all_tpr, axis=1)
    # mean_auc = np.mean(all_auc)
    # std_auc = np.std(
    #     all_auc, ddof=1
    # )  # Specify using n-1 df in oder to use same std as pandas (pandas uses sample mean)

    # ax.plot(
    #     mean_fpr,
    #     mean_tpr,
    #     color="b",
    #     label=f"Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})",
    #     lw=2,
    #     alpha=0.9,
    # )

    # # Plot standard deviation area
    # std_tpr = np.std(all_tpr, axis=1)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(
    #     mean_fpr,
    #     tprs_lower,
    #     tprs_upper,
    #     color="grey",
    #     alpha=0.2,
    #     label=f"\u00B1 1 SD",
    # )

    # # # Set axis limits
    # # ax.set(
    # #     xlim=[-0.05, 1.05],
    # #     ylim=[-0.05, 1.05],
    # # )

    # # Add title and legend
    # ax.set_title(f"ROC curve for {model_name}")
    # ax.legend()  # loc="lower right")

    # Return figure
    return fig
