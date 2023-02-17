# /usr/bin/env python
""" Common utility functions for data processing. """

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
    json_data = json.dumps(obj, sort_keys=True,
                           ignore_nan=ignore_nan, default=default)
    return md5_sha_from_str(json_data)

def prefix_dict_keys(d, s):
    """
    Prefixes the keys of a dictionary with a string
    """
    return {s+k: v for k, v in d.items()}

def split_dataframe(dataframe, key="UID", keepcols=None, train_size=0.75, random_state=None):
    """
    Split UIDs into train and test sets.
    Use the splits to get train and test dataframes.

    Args:
        dataframe: dataframe to split               
        key: column name to split on               [default: UID]
        keepcols: columns to keep in the dataframe [default: None -> all columns]
        train_size: size of test set [default: 0.75]
        random_state: random state for train_test_split [default: None]
    Returns:
        train_df, test_df
    """
    if train_size == 1.0:
        return dataframe, None
    else:
        unique_keys = dataframe[key].unique().tolist()
        if train_size < 1.0:
            train_keys, test_keys = train_test_split(
                unique_keys, train_size=train_size, random_state=random_state)
        elif train_size > 1.0:
            train_keys, test_keys = train_test_split(
                unique_keys, train_size=int(train_size), random_state=random_state)

        train_df = dataframe[dataframe[key].isin(
            train_keys)].reset_index(drop=True)
        test_df = dataframe[dataframe[key].isin(
            test_keys)].reset_index(drop=True)
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
    return 1/(1+np.exp(-X))


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
    return 1 - np.prod([1-p for p in probabilities])


def uid_aggregate(data, variables, init_suffix="", uid_column="uid", post_suffix="", agg_function="mean"):
    """
    Aggregate text level records to UID-level records.

    Args:

        `data`:
            Dataframe
        `uid_column`:
            UID identifier column
        `annotation_columns`:
            List of annotation columns for which
            aggregation will be applied at `UID` level
        `agg_function`:
            `str` type that identifies a function in numpy,
                or a callable function that transforms the data

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
        init_variables[x]: post_variables[x] for x in range(len(variables))}

    data_grouped = data[[uid_column, *init_variables]].groupby(uid_column)
    agg_groups = []
    for g, group in data_grouped:
        agg_grp = {uid_column: g}
        agg_grp.update(dict(group[init_variables].apply(agg_function, axis=0)))
        agg_groups.append(agg_grp)

    data_uid = pd.DataFrame(agg_groups)
    data_uid = data_uid.rename(columns=rename_variables)
    return data_uid


def merge_dataframes(dataframes, on="uid", how='inner'):
    """
    Merge multiple dataframes into one.
    """
    return functools.reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dataframes)


'''
# TODO:
###  Deprecated functions. To be revisited


def uid_aggregate_with_relevance(
    data,
    uid_column,
    annotation_columns,
    cluster_label_dict,
    suffix="",
    agg_function="mean",
):
    """
    Aggregate text level records to UID-level records.

    Args:
        `data`                  : Dataframe
        `uid_column`            : UID identifier column
        `annotation_columns`    : List of annotation columns for which aggregation will be applied at `UID` level
        `agg_function`          : `str` type that identifies a function in numpy, or a callable function that transforms the data

    """
    uid_list = data[uid_column].unique().tolist()
    if isinstance(agg_function, str):
        try:
            agg_function = getattr(np, agg_function)
        except:
            raise NoOptionError(
                "`agg_function` must be a numpy function or a custom callable function"
            )
    data_uid = pd.DataFrame(uid_list, columns=[uid_column])
    for c, annotation_var in enumerate(annotation_columns):
        selected_cluster_labels = cluster_label_dict[annotation_var]
        df_uid_anno = (
            data.loc[
                data.cluster_label.isin(selected_cluster_labels),
                [uid_column, annotation_var + suffix],
            ]
            .groupby(uid_column)
            .apply(agg_function)
            .reset_index()
        )
        data_uid = pd.merge(data_uid, df_uid_anno, on="uid", how="outer")
        data_uid = data_uid.fillna(0)
    return data_uid

def aggregate_variables(
    data_uid,
    uid_column,
    annotation_columns,
    suffix="",
    aggregation_params={},
):
    """
    Create aggregate variables from UID-level dataframe.
    """
    df_uid = data_uid[
        [uid_column, *[a + suffix for a in annotation_columns]]
    ].copy()

    for agg_column, agg_instructions in aggregation_params.items():

        base_columns = [a + suffix for a in agg_instructions["columns"]]
        agg_function = agg_instructions["agg_func"]
        if isinstance(agg_function, str):
            try:
                agg_function = getattr(np, agg_function)
            except:
                raise NoOptionError(
                    "`agg_func` must be a numpy function or a custom callable function"
                )
        df_uid[agg_column + suffix] = df_uid[base_columns].apply(
            agg_function, axis=1
        )
    return df_uid.copy()

class Dataset:
    """
    Base class for dataframe operations.

    """

    def __init__(
        self,
        annotated_df,
        unannotated_df,
        uid_column=None,
        text_column=None,
        annotation_columns=[],
    ):
        """
        Args:

            - dataframe
            - uid_column
            - text_column
            - annotation_columns

        """
        self.annotated_df = annotated_df.copy()
        self.unannotated_df = unannotated_df.copy()

        self.columns = self.annotated_df.columns

        assert (
            uid_column in self.columns
        ), f"The uid_column: {uid_column} does not exist in dataframe"
        self.uid_column = uid_column
        assert (
            text_column in self.columns
        ), f"The text_column:  {text_column} does not exist  in dataframe"
        self.text_column = text_column

        missing_columns = np.setdiff1d(annotation_columns, self.columns)
        assert (
            len(missing_columns) == 0
        ), f"The following annotation_columns: {';'.join(missing_columns)} do not exist in dataframe"

        self.annotation_columns = annotation_columns

        self.annotated_uids = self.annotated_df[self.uid_column].unique()
        self.unannotated_uids = self.unannotated_df[self.uid_column].unique()

        common_uids = np.intersect1d(
            self.annotated_uids, self.unannotated_uids
        )
        assert (
            len(common_uids) == 0
        ), f"Annotated and unannotated UIDs not mutually exclusive. Please check the {';'.join(common_uids)}"

        self.num_annotated_uids = len(self.annotated_uids)
        self.num_unannotated_uids = len(self.unannotated_uids)
    def set_train_test_split(self, train_pct=0.75):
        assert 0 < train_pct <= 1, "train_pct must be between 0 & 1"

        self.train_pct = train_pct
        self.test_pct = 1 - train_pct

        self.train_split_size = int(self.num_annotated_uids * self.train_pct)
        self.test_split_size = self.num_annotated_uids - self.train_split_size
        if self.test_split_size > 0:
            self.train_uuids, self.test_uuids = train_test_split(
                self.annotated_uids,
                train_size=self.train_split_size,
                test_size=self.test_split_size,
                shuffle=True,
            )
        else:
            self.train_uuids = self.annotated_uids
            self.test_uuids = []
        self.train_data = self.annotated_df[
            self.annotated_df[self.uid_column].isin(self.train_uuids)
        ].reset_index(drop=True)
        self.test_data = self.annotated_df[
            self.annotated_df[self.uid_column].isin(self.test_uuids)
        ].reset_index(drop=True)
    def get_unannotated_data(self):
        """ """
        U = self.unannotated_df[self.uid_column].values
        X = self.unannotated_df[self.text_column].values
        return U, X
    def get_annotated_data(self):
        """ """
        U = self.annotated_df[self.uid_column].values
        X = self.annotated_df[self.text_column].values
        return U, X
    def get_train_data(self, annotation_var=None, na_fill=0):
        """ """
        X = self.train_data[self.text_column].values
        assert (
            annotation_var in self.annotation_columns
        ), f"{annotation_var} not in annotation_columns"
        Y = self.train_data[annotation_var].replace(np.nan, na_fill).values

        return X, Y
    def get_test_data(self, annotation_var=None, na_fill=0):
        """ """
        X = self.test_data[self.text_column].values
        assert (
            annotation_var in self.annotation_columns
        ), f"{annotation_var} not in annotation_columns"
        Y = self.test_data[annotation_var].replace(np.nan, na_fill).values

        return X, Y
    def get_train_test_uuids(self):
        uuid_assignments = {s: "train" for s in self.train_uuids}
        uuid_assignments.update({s: "test" for s in self.test_uuids})
        return uuid_assignments
    def aggregate_uid(self, data, uid_agg_func="mean", aggregation_params={}):
        """ """
        self.aggregation_params = aggregation_params
        self.uid_agg_func = uid_agg_func
        data_uid = uid_aggregate(
            data,
            self.uid_column,
            self.annotation_columns,
            agg_function=uid_agg_func,
        )
        if len(self.aggregation_params) > 0:
            data_uid = aggregate_variables(
                data_uid,
                uid_column=self.uid_column,
                annotation_columns=self.annotation_columns,
                aggregation_params=self.aggregation_params,
            )
        return data_uid.copy()
    def aggregate_annotated_dataset(
        self, uid_agg_func="mean", aggregation_params={}
    ):
        self.annotated_uid_df = self.aggregate_uid(
            self.annotated_df,
            uid_agg_func=uid_agg_func,
            aggregation_params=aggregation_params,
        )
        return self.annotated_uid_df.copy()
'''
