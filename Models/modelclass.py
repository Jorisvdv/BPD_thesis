"""
Protocol for model object

@Author: Joris van der Vorst <joris@jvandervorst.nl>
@License: MIT
@Date: 2023-09-19

"""

# <codecell> Packages
# Import packages
from typing import Protocol, Union

from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor

# <codecell> Dataclass
# Define dataclass
# TODO: Add dataclass for model


# <codecell> Model class
# Define model class
class Model(Protocol):
    def fit(
        self,
        X: Union[DataFrame, ndarray, Tensor],
        y: Union[DataFrame, Series, ndarray, Tensor],
    ) -> None:
        ...

    def predict(
        self, X: Union[DataFrame, Series, ndarray, Tensor]
    ) -> Union[DataFrame, ndarray, Tensor]:
        ...

    def predict_proba(
        self, X: Union[DataFrame, Series, ndarray, Tensor]
    ) -> Union[DataFrame, ndarray, Tensor]:
        ...
