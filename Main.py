"""
Main function to run analysis for the prediction of BPD
@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-06-27

"""
# <codecell> Packages
# Import packages
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

# <codecell> Utilities
# Import utilities
from Scripts_Joris.dataloader import load_static
from Scripts_Joris.utilities import (  # load_model,
    create_roc_curve,
    export_model_and_scores,
    nested_CV,
    print_scores,
)

# <codecell> Settings
# Script settings

MULTI_CLASS_LR = "ovr"
MAX_ITER_LR = 10000

# Define parameter search space
log_reg_parameters = {"penalty": ["l2"]}  # , "l1", "elasticnet"]}


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # <codecell> Print config
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # <codecell> Import data
    # Import data

    # Import data from csv file using pathlib to specify path relative to script
    # find location of script and set manual location for use in ipython

    # # Read csv file
    # data_static = load_static(cleaned_data=True)

    # Load iris dataset as toy example for local testing
    from sklearn.datasets import load_iris

    data_static = load_iris(as_frame=True).frame
    # Select only two classes

    # <codecell> Data selection
    # Show column names
    print(" Column names: ")
    print(*data_static.columns)
    print("Selected parameters: ")
    print(*cfg.model_parameters.selected_features)

    # Split data into features and target
    X = data_static.loc[:, cfg.model_parameters.selected_features]  # .drop("y", axis=1)
    y = data_static[cfg.model_parameters.target].where(data_static["target"] == 1, 0)
    # FIXME: This is a hack to make the iris dataset work with the current code

    # <codecell> Create Logistic regression

    # Create logistic regression model
    log_reg = LogisticRegression(
        multi_class=cfg.model_parameters.multi_class_lr,
        max_iter=cfg.model_parameters.max_iter_lr,
    )

    # <codecell> Nested cross validation
    # Nested cross validation

    models_and_scores = nested_CV(
        model=log_reg,
        parameters=OmegaConf.to_container(cfg.model_parameters.grid_search),
        inner_cv=5,
        outer_cv=5,
        X=X,
        y=y,
        verbose=0,
        random_state=cfg.train_settings.random_state,
    )

    # <codecell> Print Scores
    # Print Scores

    print_scores(models_and_scores)

    # <codecell> Plot ROC curve
    # Plot ROC curve

    fig_test = create_roc_curve(
        model_scores=models_and_scores, model_name=cfg.model_parameters.name_model
    )
    # plt.show()
    # fig_test.savefig("roc_curve_test.png")

    # <codecell> Export model and metrics
    # Export model
    if cfg.train_settings.model_export:
        export_model_and_scores(
            name_model=cfg.model_parameters.name_model,
            models_and_scores=models_and_scores,
            save_model=True,
            save_scores=True,
            # selected_parameters=SELECTED_PARAMETERS, # Can get this directly from model
        )


if __name__ == "__main__":
    main()
