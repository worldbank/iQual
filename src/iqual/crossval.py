# /usr/bin/env python
"""This script contains cross-validation methods and classes used by the iQual package"""

import operator
import warnings
import functools
import pandas as pd
import numpy as np
import sklearn.model_selection
from collections import Counter


def convert_dict(d, prefix):
    """
    Convert a nested dictionary to a flat dictionary
    """

    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            prefix_fmt = f"{prefix}__{k}".strip("__")
            result.update(convert_dict(v, prefix_fmt))
        elif isinstance(v, list):
            prefix_fmt = f"{prefix}__{k}".strip("__")
            result[prefix_fmt] = v
    return result


def convert_nested_params(configuration):
    """
    Convert a nested CV configuration to a flat dictionary
    """

    return convert_dict(configuration, "")


def count_hyperparameters(param_grid):
    """
    Count hyperparameter grid of type list-or-dict
    """

    def count_dict_params(param_dict):
        """
        Count hyperparameter grid of type dict
        """
        num_configurations = functools.reduce(
            operator.mul, [len(p) for p in param_dict.values()], 1
        )
        return num_configurations

    if isinstance(param_grid, dict):
        num_params = count_dict_params(param_grid)
    elif isinstance(param_grid, list):
        num_params = sum([count_dict_params(pdict) for pdict in param_grid])
    else:
        num_params = None
    return num_params


class CrossValidator:
    """
    Cross-validation methods
    """

    def __init__(
        self,
        model_pipe,
        search_parameters,
        cv_method="GridSearchCV",
        scoring="f1",
        cv=3,
        refit=False,
        **kwargs,
    ):
        """
        Needs a valid `cv_method` from `sklearn.model_selection`
        Configure the cross-validation object
        """

        self.cross_validation = getattr(sklearn.model_selection, cv_method)
        self.cross_validation = self.cross_validation(
            model_pipe,
            search_parameters,
            scoring=scoring,
            cv=cv,
            refit=refit,
            **kwargs,
        )

    def fit(self, X, Y, **kwargs):
        """
        Fit the cross-validation object
        """

        self.cross_validation.fit(X, Y, **kwargs)
        self.refit_string = self.cross_validation.refit
        self.n_splits = self.cross_validation.n_splits_
        if not self.refit_string:
            self.refit_string = "score"
        self.rank_column = "rank_test_" + self.refit_string
        self.best_parameters = self.get_best_params()
        self.param_df = self.get_cv_results()

        self.cv_scores = self.param_df.loc[
            self.param_df[self.rank_column] == 1,
            [f"split{i}_test_{self.refit_string}" for i in range(self.n_splits)],
        ].to_dict("records")[0]
        avg_cv_score = np.mean(list(self.cv_scores.values()))
        self.cv_scores.update({"avg_test_score": avg_cv_score})

    def get_cv_scores(self):
        """
        Get the cross-validation scores
        """
        return self.cv_scores.copy()

    def get_cv_results(self):
        """
        Get the cross-validation results as a dataframe
        """
        self.param_df = pd.DataFrame(self.cross_validation.cv_results_)
        self.param_df = self.param_df.sort_values(
            by=self.rank_column, ignore_index=True
        )
        return self.param_df.copy()

    def predict(self, X):
        """
        Predict using the best estimator
        """
        return self.cross_validation.predict(X)

    def get_best_params(self):
        """
        Get the best parameters
        """
        return self.cross_validation.best_params_


class Splitter:
    """
    Customized CV Splitter class.

    To ensure that a minimum threshold of positive class is satisfied for the split

    """

    def __init__(self, method: str, thresh=0.2, attempts=3, **kwargs) -> None:
        """
        Initialize the splitter object.
        """
        self.method = method
        self.thresh = thresh
        self.valid_split = True
        self.attempts = attempts
        self.kwargs = kwargs
        self.splitter = getattr(sklearn.model_selection, method)(**kwargs)

    def get_n_splits(self, X, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        """
        return self.splitter.get_n_splits(X, y, groups)

    def split_and_verify(self, X, y=None, groups=None) -> list:
        """
            NOTE: Threshold: 20% of total positives

        Split the data and verify that the minimum threshold of positive class is satisfied

        split_list: list of tuples of (train_index, test_index)
        """
        total_positives = Counter(y)[1]
        n_splits = self.splitter.get_n_splits(X, y, groups=groups)
        if total_positives < n_splits:
            self.valid_split = False
        else:
            self.valid_split = True
            split_generator = self.splitter.split(X, y, groups=groups)
            split_list = [next(split_generator) for _ in range(n_splits)]
            for n in range(n_splits):
                y_values = [y[j] for j in split_list[n][1]]
                y_count = Counter(y_values)[1]
                pct_thresh = y_count / total_positives
                if pct_thresh > self.thresh:
                    continue
                else:
                    self.valid_split = False
                    break
            return iter(split_list)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        for _ in range(self.attempts):
            split_generator = self.split_and_verify(X, y, groups)
            if self.valid_split:
                break
            else:
                continue
        if not self.valid_split:
            warnings.warn("Splitter Threshold not met")
        return split_generator

    def get_datasets(self, X, y=None):
        """
        Get the datasets for each split
        """
        split_list = self.split(X, y)
        datasets = []
        for train_index, test_index in split_list:
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            datasets.append(
                {
                    "train_X": X_train,
                    "test_X": X_test,
                    "train_y": y_train,
                    "test_y": y_test,
                }
            )
        return datasets
