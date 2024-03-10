"""
Specification of regression model pipeline
@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-09-19

"""

# <codecell> Packages
# Import packages
from typing import Any, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# <codecell> model creation


def make_gridsearch(
    classifier: Union[LogisticRegression, RandomForestClassifier, SVC],
    param_grid: dict[str, Any],
    verbose: int = 0,
    cv: int = 5,
    useFeatureSelection: bool = True,
    useScaler: bool = True,
) -> Pipeline:
    """
    Create a pipeline for a given model class and parameter grid that uses recursive feature elimination

    Parameters
    classifier : Union[LogisticRegression, RandomForestClassifier, SVC]: model used for classification
    param_grid : dict[str, Any]: parameter grid for the model
    verbose : int, optional: verbosity level. Defaults to 0.
    cv : int, optional: number of folds for cross-validation. Defaults to 5.
    useFeatureSelection: bool, optional: whether to use feature selection. Defaults to True.
    useScaler: bool, optional: whether to use an scaler. Defaults to True.
    """

    scaler = StandardScaler()
    selector = RFECV(
        estimator=classifier,
        step=1,
        cv=cv,
        scoring="roc_auc",
        verbose=verbose,
    )

    # Build pipeline
    #
    modelsteps = []
    if useScaler:
        modelsteps.append(("scaler", scaler))
    if useFeatureSelection:
        modelsteps.append(("feature_selection", selector))
    else:
        modelsteps.append(("estimator", classifier))

    pipeline = Pipeline(modelsteps)

    # gridsearch = GridSearchCV(
    #     pipeline,
    #     param_grid=param_grid,
    #     cv=cv,
    #     refit="roc_auc",
    #     scoring=["accuracy", "roc_auc", "f1"],
    #     n_jobs=-1,
    #     verbose=verbose,
    # )
    # return gridsearch
    return pipeline
