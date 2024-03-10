"""
Class for cross valiation loop for ML models

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-10-09

"""

import logging
from pathlib import Path
from typing import List, Union

from joblib import dump, load
from Models.modelclass import Model
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from Utils.metrics import Metrics, calculateMetrics_from_model


class CrossValidation:
    """
    Class for cross validation of a model.

    Parameters:
    n_splits (int): The number of splits to use in the cross validation.
    random_state (int): The random state to use in the cross validation.
    dataset (Union[DataFrame, ndarray]): The dataset object to use for cross validation.
    cv_file (Path, optional): The Path to a pickle file containing pre-generated cross validation folds.
    logger (Union[None, logging.Logger]): The logger object to use for logging.
    """

    def __init__(
        self,
        n_splits: Union[int, None] = None,
        random_state: Union[int, None] = None,
        X: Union[DataFrame, ndarray, None] = None,
        y: Union[DataFrame, Series, ndarray, None] = None,
        cv_file: Union[Path, None] = None,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_file = cv_file
        self.X = X
        self.y = y
        self.skf = None
        self.cv_folds = None

        if cv_file is not None:
            # Ensure the file exists
            if not cv_file.exists():
                raise FileNotFoundError(
                    f"Cross validation file {cv_file} does not exist."
                )
            self.load_cv_folds()
        else:
            if self.X is None or self.y is None:
                raise ValueError(
                    "Either a cv_file or X and y must be provided to generate cross validation folds."
                )
            self.generate_cv_folds(self.X, self.y)

    def load_cv_folds(self) -> None:
        self.cv_folds = load(self.cv_file)

    def save_cv_folds(self, filename: Union[str, Path]) -> None:
        dump(self.cv_folds, filename)

    def generate_cv_folds(
        self, X: Union[DataFrame, ndarray], y: Union[DataFrame, Series, ndarray]
    ) -> None:
        """
        Generate cross validation fold indices.
        """
        if self.n_splits is None:
            raise ValueError("n_splits must be provided to generate cross validation.")
        self.skf = StratifiedKFold(
            n_splits=self.n_splits, random_state=self.random_state, shuffle=True
        )
        self.cv_folds = list(self.skf.split(X, y))
        if self.cv_file is not None:
            self.save_cv_folds(self.cv_file)

    def cross_validate(
        self,
        model: Model,
        X: DataFrame,
        y: Union[DataFrame, Series],
    ) -> List[Metrics]:
        """
        Cross validate a model using the previously generated or newly generated cross validation folds.

        Parameters:
        model (modelclass.Model): The model object to use for prediction.
        X (DataFrame): The feature data to use for prediction.
        y (Union[DataFrame, Series]): The target data to use for prediction.

        Returns:
        List[(Metrics, modelclass.Model)]: A list of tuples containing the metrics and the trained model object.
        """
        if self.cv_folds is None:
            self.generate_cv_folds(X, y)

        cv_results = []
        if self.cv_folds is None:
            raise ValueError("Cross validation folds have not been generated.")

        for fold, (train_index, test_index) in enumerate(self.cv_folds):
            print(f"Fold {fold + 1}/{len(self.cv_folds)}")
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)

            metrics: Metrics = calculateMetrics_from_model(model, X_test, y_test)

            logging.info(
                "Fold %s: Accuracy: %s, ROC AUC: %s, F1: %s",
                fold + 1,
                metrics.accuracy,
                metrics.roc_auc,
                metrics.f1,
            )
            # Check if model is a GridSearchCV object
            if isinstance(model, GridSearchCV):
                # Log best parameters
                # Check if model uses feature selection
                if hasattr(model.best_estimator_.named_steps, "feature_selection"):
                    # Log best features
                    logging.info(
                        "Best features: %s",
                        model.feature_names_in_[
                            model.best_estimator_.named_steps[
                                "feature_selection"
                            ].support_
                        ],
                    )
            # Log best parameters in pipeline
            if isinstance(model, Pipeline):
                # Log best parameters
                # Check if model uses feature selection
                if hasattr(model.named_steps, "feature_selection"):
                    # Log best features
                    logging.info(
                        "Best features: %s",
                        model.feature_names_in_[
                            model.named_steps["feature_selection"].support_
                        ],
                    )

            cv_results.append(metrics)

        return cv_results
