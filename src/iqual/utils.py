# /usr/bin/env python
"""Utility functions"""

import pickle
import hashlib
import functools
import numpy as np
import pandas as pd
import simplejson as json
from configparser import NoOptionError
from sklearn.model_selection import train_test_split


def md5_sha_from_str(val: str) -> str:
    """
    Generate md5 and sha256 hash from string.
    """
    return hashlib.md5(val.encode("utf-8")).hexdigest()


def md5_sha_from_dict(obj, ignore_nan=False, default=None):
    """
    Generate md5 and sha256 hash from dictionary.
    """
    json_data = json.dumps(obj, sort_keys=True, ignore_nan=ignore_nan, default=default)
    return md5_sha_from_str(json_data)


def prefix_dict_keys(d, s):
    """
    Prefixes the keys of a dictionary with a string
    """
    return {s + k: v for k, v in d.items()}


def split_dataframe(
    dataframe, key="UID", keepcols=None, train_size=0.75, random_state=None
):
    """
    Split UIDs into train and test sets.
    Use the splits to get train and test dataframes.

    Args:
        dataframe (pd.DataFrame): dataframe to split
        key (str): column name to split on [default: UID]
        keepcols (list): list of columns to keep in train and test dataframes [default: None]
        train_size (float): proportion of data to use for training [default: 0.75]
        random_state (int): random seed [default: None]

    Returns:
        train_df (pd.DataFrame): train dataframe
        test_df (pd.DataFrame): test dataframe
    """
    if train_size == 1.0:
        return dataframe, None
    else:
        unique_keys = dataframe[key].unique().tolist()
        if train_size < 1.0:
            train_keys, test_keys = train_test_split(
                unique_keys, train_size=train_size, random_state=random_state
            )
        elif train_size > 1.0:
            train_keys, test_keys = train_test_split(
                unique_keys, train_size=int(train_size), random_state=random_state
            )
        train_df = dataframe[dataframe[key].isin(train_keys)].reset_index(drop=True)
        test_df = dataframe[dataframe[key].isin(test_keys)].reset_index(drop=True)
        if keepcols is not None:
            train_df = train_df[keepcols]
            test_df = test_df[keepcols]
        return train_df, test_df


def pad_left(x, n=3):
    """
    Converts strings/ integers to 3-digit string code.
    """
    x = str(int(x))
    while len(x) < n:
        x = "0" + x
    return x


def combine_dataframes(dataframes):
    """
    Combine multiple dataframes into one.
    """
    data = pd.concat(dataframes, ignore_index=True)
    return data


def get_probas_from_decisions(X):
    """
    Reverse decision function values to get \
        pseudo-probabilities between 0 & 1

    Args:
        X: decision function output

    Returns: pseudo-probabilities between 0 & 1
        dtype: float
    """
    return 1 / (1 + np.exp(-X))


def save_pickle_data(doc, pkl_filepath):
    """
    Save pickle data to file.
    """
    with open(pkl_filepath, "wb") as pkl_file:
        pickle.dump(doc, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved file in {pkl_filepath}")


def load_pickle_data(pkl_filepath):
    """
    Load pickle data from file.
    """
    with open(pkl_filepath, "rb") as pkl_file:
        doc = pickle.load(pkl_file)
    return doc


def save_json_data(doc, json_filepath):
    """
    Save json data to file.
    """
    with open(json_filepath, "w") as json_file:
        json.dump(doc, json_file)
    print(f"Saved file in {json_filepath}")


def load_json_data(json_filepath):
    """
    Load json data from file.
    """
    with open(json_filepath, "r") as json_file:
        doc = json.load(json_file)
    return doc


def probability_atleast_1(probabilities):
    """
    Calculate the probability of atleast 1 success from a list of probabilities of successes
    """
    return 1 - np.prod([1 - p for p in probabilities])


def uid_aggregate(
    data,
    variables,
    init_suffix="",
    uid_column="uid",
    post_suffix="",
    agg_function="mean",
):
    """
    Aggregate text level records to UID-level records.

    Args:
        data (pd.DataFrame): Dataframe with text level records
        variables (list): List of variables to aggregate
        init_suffix (str): Suffix to add to variables before aggregation
        uid_column (str): UID column name
        post_suffix (str): Suffix to add to variables after aggregation
        agg_function (str): Aggregation function to use

    Returns:
        pd.DataFrame: Dataframe with UID level records
    """
    if isinstance(agg_function, str):
        try:
            agg_function = getattr(np, agg_function)
        except Exception as e:
            raise NoOptionError(
                "`agg_function` must be a numpy function or a custom callable function"
            )
    init_variables = [a + init_suffix for a in variables]
    post_variables = [a + post_suffix for a in variables]
    rename_variables = {
        init_variables[x]: post_variables[x] for x in range(len(variables))
    }

    data_grouped = data[[uid_column, *init_variables]].groupby(uid_column)
    agg_groups = []
    for g, group in data_grouped:
        agg_grp = {uid_column: g}
        agg_grp.update(dict(group[init_variables].apply(agg_function, axis=0)))
        agg_groups.append(agg_grp)
    data_uid = pd.DataFrame(agg_groups)
    data_uid = data_uid.rename(columns=rename_variables)
    return data_uid


def merge_dataframes(dataframes, on="uid", how="inner"):
    """
    Merge multiple dataframes into one.
    """
    return functools.reduce(
        lambda left, right: pd.merge(left, right, on=on, how=how), dataframes
    )