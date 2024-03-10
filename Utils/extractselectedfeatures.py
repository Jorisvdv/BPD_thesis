import pandas as pd

from Models.modelclass import Model
from Utils.metrics import Metrics


def extract_selected_features(
    output: list[tuple[Metrics, Model]], X: pd.DataFrame
) -> list[str]:
    """
    Extract the selected features from the output of the cross validation
    """
    selected_features_list = []
    for loop in output:
        # Access the fitted RFE selector from the best estimator
        selected_features_mask = (
            loop[1].best_estimator_.named_steps["feature_selection"].support_
        )

        # Print the selected features
        selected_features = str(X.columns[selected_features_mask])
        selected_features_list.append(selected_features)
    return selected_features_list
