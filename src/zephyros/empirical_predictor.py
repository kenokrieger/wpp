"""
Module for empirical prediction of the estimated power output of a wind turbine.
"""
# Copyright (C) 2024  Keno Krieger <kriegerk@uni-bremen.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from copy import deepcopy

import numpy as np
import pandas as pd

PREDICT_KEY = "predicted"
UNCERTAINTY_KEY = "uncertainty"


def learn_and_predict(learn_data, predict_data, features, target, accuracy=3):
    """
    Convenience function that combines the zephyros.empirical_predictor.learn
    and zephyros.empirical_predictor.predict methods.

    Given *learn_data* and *predict_data*, predict values for the *target*
    based on historically measured values.

    Args:
        learn_data(pd.DataFrame): Historical data used for "learning".
        predict_data(pd.DataFrame): Data containing features used for predicting
            the target variable.
        features(list): Column names of the features. The order is important as
            it determines which feature is used for which split, e.g. the first
            feature is used for the first split and so on.
        target(str): Column name of the target variable of the prediction.
        accuracy(int): The number of splits to perform for each feature.

    Returns:
        pd.DataFrame: The predicted value(s).
    """
    learned = learn(learn_data, features, target, accuracy)
    return predict(learned, predict_data)


def predict(learned, x):
    """

    Args:
        learned (pd.DataFrame): A pandas DataFrame with an IntervalIndex
            containing the mean of historically measured values in the given
            interval(s).
        x(pd.DataFrame): Data containing the features used for the
            prediction.

    Returns:
        pd.DataFrame: The predicted value(s).

    """
    features = learned.index.names
    target = list(learned.keys())
    _catch_missing_keys(x, features)
    prediction = x.apply(_predict_row, axis="columns",
                         args=(learned, features, target))
    return pd.DataFrame(prediction.to_list(), columns=target,
                        index=prediction.index)


def learn(x, features, target, accuracy=3):
    """
    Create a multiindex DataFrame that contains the average value of *target*
    for intervals of each feature from *features*. The accuracy determines the
    number of intervals/splits for each feature.

    Args:
        x(pd.DataFrame): A DataFrame containing columns with *features* and a
            *target* variable.
        features(list): Column names of the features. The order is important as
            it determines which feature is used for which split, e.g. the first
            feature is used for the first split and so on.
        target(str): The column name of the target variable.
        accuracy(int): The number of intervals for each feature.

    Returns:
        pd.DataFrame: A multiindex DataFrame containing the target variable
            under the key 'predicted' and its uncertainty under 'uncertainty'.

    """
    _catch_accuracy_to_high(x, features, accuracy)
    paths, path_values = grow_tree(x, iter(features), target, accuracy)
    multi_index = pd.MultiIndex.from_tuples(paths, names=features)
    learned = pd.DataFrame(data=path_values,
                           columns=[PREDICT_KEY, UNCERTAINTY_KEY],
                           index=multi_index)
    if learned.isnull().values.any():
        print("WARNING: Resulting dataframe contains NaN values. This might "
              "be due to a fine splitting, i.e. high accuracy or many "
              "features.")
    return learned


def _catch_accuracy_to_high(x, features, accuracy):
    """
    Check if the desired accuracy in combination with the number of features
    can be achieved and raises a ValueError if the combination is impossible.

    Args:
        x(pd.DataFrame): The DataFrame to serve as data for the learning
            process.
        features(list): The features used for the learning.
        accuracy(int): The requested accuracy, i.e. number of splits.

    Raises:
        ValueError: If the requested combination of accuracy and number of
            features is impossible to achieve.

    """
    nrows = x.shape[0]
    fragmentation = accuracy ** len(features)

    if fragmentation >= nrows:
        raise ValueError(f"The request accuracy ({accuracy}) in combination " 
                         f"with the number of features ({len(features)})"
                         "is to high!\n"
                         f"pow(accuracy, len(features) = {fragmentation} "
                         f"needs to be smaller than the number of rows in the "
                         f"data x.shape[0] = {nrows}")


def _catch_missing_keys(x, required):
    """
    Check if all necessary keys for the calculations are present and, if not,
    raise a key error specifying which keys are missing.

    Args:
        x(pd.DataFrame): Parameters required for the prediction of
            the power output.

    Returns:
        None.

    Raises:
        KeyError: If any of the required keys are missing.

    """
    missing = []

    for key in required:
        if key not in x:
            missing.append(key)

    if missing:
        err_msg = "Missing required keys for calculation: "
        err_msg += ", ".join(missing)
        raise KeyError(err_msg)


def _myloc(learned, feature_values):
    """
    Custom implementation of pandas DataFrame.loc method. Indexing a Multiindex
    containing IntervalIndices has known bugs. This implementation circumvents
    using the DataFrame.loc method.

    Iteratively filter the index of *learned* by selecting the intervals
    containing the combination of *feature_values*. If the values are out of
    bound map them to the nearest interval instead. Return the combination
    of intervals containing *feature_values*.

    Args:
        learned(pd.DataFrame): The DataFrame to index.
        feature_values(iterable): Values to search for at each index level
            respectively.

    Returns:
        tuple: The intervals containing the matching *feature_values*.

    """
    mask = learned.index
    for i, fv in enumerate(feature_values):
        new_mask = [mi for mi in mask if fv in mi[i]]
        if not new_mask:
            new_mask = _handle_out_of_bounds(fv, i, mask)
        mask = new_mask
    return mask[0]


def _predict_row(row, learned, features, target):
    """
    Predict the value for the *target* variable given a set of features by using
    binned historically found values for the *target* given the *features*.

    Args:
        row (pd.Series): A pandas Series containing features used for the
            prediction.
        learned(pd.DataFrame): Historically measured values binned in discrete
            intervals. If a feature is out of bound of the historic values, use
            the nearest existing interval.
        features(list): A list of column names to use for the prediction.
        target(str): The column name of the desired prediction value.

    Returns:
        list: The predicted value(s) given the features.

    """
    feature_values = tuple(row[f] for f in features)
    # tuple indexing (learned.loc[feature_values]) does not work if values are
    # out of bounds of the intervals
    # this iterative approach works more reliably
    index = _myloc(learned, feature_values)
    for loc in index:
        learned = learned.loc[loc]
    return learned.values.reshape((2, ))


def _handle_out_of_bounds(fv, level, mask):
    """
    Projects out of bounds feature value *fv* to the nearest existing interval.
    Args:
        fv(float): The value of a feature.
        level(int): Level of the MultiIndex.
        mask(list): A list containing IntervalIndices.

    Returns:
        list: The *mask* with the interval closest to *fv*.

    """
    lower_bound = np.min([mi[level] for mi in mask])
    upper_bound = np.max([mi[level] for mi in mask])
    if _out_of_bounds_left(fv, lower_bound):
        mask = [mi for mi in mask if mi[level] == lower_bound]
    else:
        mask = [mi for mi in mask if mi[level] == upper_bound]
    return mask


def _out_of_bounds_left(v, interval):
    """
    Check whether a value is smaller than the left bound of an interval.

    Args:
        v(float): The value to perform the check for.
        interval(pd.Interval): The interval to perform the check for.

    Returns:
        bool: True if the interval is smaller than the left bound of the
            interval, False otherwise.
    """
    return v < interval.left


def grow_tree(x, by, target, accuracy):
    """
    Convenience function for zephyros.empirical_predictor._grow_tree.

    Given column names '*by*', group data in a DataFrame recursively by those
    column's values and return the index and the mean and standard deviation
    of the target variable.

    Args:
        x (pd.DataFrame): The DataFrame to group.
        by (iterable): An iterator of column names.
        target (str): Name of the target column.
        accuracy (int): The number of splits for each column.

    Returns:
        tuple: The index and mean and standard deviation of the target variable.

    """
    return _grow_tree(x, by, target, accuracy, [], [], [])


def _grow_tree(x, by, target, accuracy, path, paths_out, values_out):
    """
    Given column names '*by*', group data in a DataFrame recursively by those
    column's values and return the index and the mean and standard deviation
    of the target variable.

    Args:
        x (pd.DataFrame): The DataFrame to group.
        by (iterable): An iterator of column names.
        target (str): Name of the target column.
        accuracy (int): The number of splits for each column.
        path (list): Dummy variable to store the index of the resulting variable.
        paths_out (list): Dummy variable to store all paths.
        values_out (list): Dummy variable to store all values.

    Returns:
        tuple: The index and mean and standard deviation of the target variable.

    """
    try:
        next_by = next(by)
    except StopIteration:
        paths_out.append(path)
        values_out.append((x[target].mean(), x[target].std()))
    else:
        new_x = _split_dataframe(x, next_by, accuracy)
        for label, group in new_x:
            _grow_tree(group, deepcopy(by), target, accuracy, path + [label],
                       paths_out, values_out)
    return paths_out, values_out


def _split_dataframe(x, by, accuracy):
    """
    Split a DataFrame *x* into a group of equally sized dataframes based on
    values of a column "*by*". The accuracy determines the number of dataframes
    returned.

    Args:
        x(pd.DataFrame): The DataFrame to slice into equally sized groups.
        by(str): The name of the column to slice by.
        accuracy(int): The number of groups to create.

    Returns:
        pd.DataFrameGroupBy: The equally sized groups.

    """
    cut = pd.qcut(x[by], accuracy, duplicates="drop")
    return x.groupby(cut, observed=True)
