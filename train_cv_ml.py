"""
Main function to run analysis for the prediction of BPD
@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-27

"""


# <codecell> Packages
# Import packages
import logging
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataloader import load_static
from Models.modelclass import Model

# import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from Utils.crossvalidation import CrossValidation
from Utils.extractselectedfeatures import extract_selected_features
from Utils.makegridvc import make_gridsearch
from Utils.metrics import (
    Metrics,
    calculateMetrics_from_model,
    plot_and_calculate_metrics,
)

# <codecell> Settings

logger: logging.Logger = logging.getLogger()


CV_FILE = Path("processed_data/cv_folds.pkl")
# METRICS_FOLDER = Path("metrics")


from hydra import compose, initialize
from omegaconf import OmegaConf

# with initialize(version_base=None, config_path="conf/"):
#     cfg = compose(config_name="config_ml")
#     print(OmegaConf.to_yaml(cfg))


# <codecell> main function
@hydra.main(config_path="conf", config_name="config_ml", version_base=None)
def main(cfg: DictConfig) -> None:
    # def main():
    # <codecell> data

    # Select features and target
    # X = data_DF.drop(columns=["bpd"])
    # y = data_DF["bpd"]
    # Load in dataframe

    data_DF = load_static(cleaned_data=cfg.dataset.cleaned_name)

    # iris = load_iris()
    # df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # df["target"] = iris.target

    # Select features and target
    # X = df.drop(columns=["target"])
    # X: pd.DataFrame = df.loc[:, cfg.dataset.selected_features]
    # y = (df["target"] == cfg.dataset.target).astype(int)
    X: pd.DataFrame = data_DF.loc[:, cfg.dataset.selected_features]
    y: pd.Series = data_DF[cfg.dataset.target]

    logger.info(
        "Patients with missing data that is imputed: %s", X.isna().any(axis=1).sum()
    )
    # Replace missing values with median of column
    X = X.fillna(X.median())

    # Set up cross validation
    if CV_FILE.exists():
        # Load cross validation folds
        cvData = CrossValidation(cv_file=CV_FILE)
    else:
        # Generate cross validation folds
        cvData = CrossValidation(
            X=X,
            y=y,
            n_splits=cfg.outer_cv,
            random_state=cfg.random_state,
        )
        # Save cross validation folds
        cvData.save_cv_folds(filename=CV_FILE)

    # <codecell> model

    # create model
    model = hydra.utils.instantiate(cfg.model.model)
    # model = RandomForestClassifier()
    # model = SVC(probability=True)

    # Change gridsearch parameters names for estimator in RFE
    parameters = dict(cfg.model.param_grid)
    if cfg.feature_selection:
        prepend = "feature_selection__"
        parameters = {prepend + key: value for key, value in parameters.items()}

    # Set up model pipeline
    pipeline = make_gridsearch(
        model,
        param_grid=parameters,
        verbose=1,
        cv=cfg.inner_cv,
        useFeatureSelection=cfg.feature_selection,
        useScaler=cfg.scaler,
    )

    # <codecell> run
    # Run cross validation

    output: list[Metrics] = cvData.cross_validate(
        model=pipeline,
        X=X,
        y=y,
    )

    # # <codecell> save output
    plot_and_calculate_metrics(
        outputs=output,
        folder=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        model_name=cfg.model.name_model,
        dataset_name=cfg.dataset.name,
    )


# <codecell> run
if __name__ == "__main__":
    # Run main function
    main()
