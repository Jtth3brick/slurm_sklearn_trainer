import copy
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

SPLIT_DIR = "split_data"


@dataclass(frozen=True)
class SplitConfig:
    """
    Contains data to train a model on.

    Attributes:
        split_id (str): Identifier for the data split.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (Optional[pd.DataFrame]): Validation features, if validation is enabled. Assumed to have same columns as X_train.
        y_val (Optional[pd.Series]): Validation labels, if validation is enabled.
        X_cv (pd.DataFrame): CV feature, if validation is enabled. Assumed to have same columns as X_train.
        X_cv (pd.Series): CV labels.
        cv_indices (List[np.ndarray]): Contains the splits for cross validation.
            May contain indices from train and validation
            None if cv is False.
    """

    split_id: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.Series]
    X_cv: Optional[pd.DataFrame]
    y_cv: Optional[pd.Series]
    cv_indices: Optional[List[np.ndarray]]


@dataclass
class ModelConfig:
    """
    Contains configuration required for training and evaluating a model.

    Attributes:
        split_id (str): Identifier for the data split.
        pipeline_name (str): Name of the training pipeline (e.g., 'enet', 'rf').
        _unfit_pipe (Pipeline): The unfit pipeline
        cv_scores (List[float]): Cross-validation AUC scores.
        validate_score (float): Optional validation AUC score.
        config_hash (str): Unique identifier generated based on configuration.
    """

    # set by manager
    split_id: str
    pipeline_name: str
    _unfit_pipe: Pipeline

    # set by worker after fitting
    model: Optional[Pipeline] = field(
        default=None, repr=False
    )  # preferrably store fitted models elsewhere
    cv_scores: List[float] = field(default_factory=list, repr=False)
    validate_score: Optional[float] = None

    _config_hash: Optional[str] = None

    @property
    def config_hash(self) -> str:
        """
        Returns the config hash.
        """
        if self._config_hash is None:
            self._config_hash = self.create_hash()
        return self._config_hash

    def create_hash(self) -> str:
        """
        Uses pipeline_hyperparameters and split_name.
        """
        hyperparams = self._unfit_pipe.get_params()
        hyperparams_str = str(sorted(hyperparams.items()))
        unique_str = f"{hyperparams_str}|split_data_id:{self.split_id}"
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def get_empty_pipe(self) -> Pipeline:
        """Returns a fresh copy of the pipeline with hyperparameters set"""
        return copy.deepcopy(self._unfit_pipe)

    def __str__(self):
        return (
            f"ModelConfig(\n"
            f"  Config Hash: {self.config_hash}\n"
            f"  Split Name: {self.split_id}\n"
            f"  Pipeline Name: {self.pipeline_name}\n"
            f"  Hyperparameters: {self._unfit_pipe.get_params()}\n"
            f"  Val Score: {self.validate_score}\n"
            f")"
        )
