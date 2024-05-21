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
import pandas as pd
from copy import deepcopy


def learn(x, features, target, accuracy=3):
    paths, values = grow_tree(x, iter(features), target, accuracy)
    print(paths)
    multi_index = pd.MultiIndex.from_tuples(paths, names=features)
    return pd.DataFrame(data=values, columns=["power", "delta_p"],
                        index=multi_index)


def grow_tree(x, by, target, accuracy):
    """
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
    return _grow_tree(x, by, target, accuracy)


def _grow_tree(x, by, target, accuracy, path=[], paths_out=[], values_out=[]):
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
        new_x = slice_dataframe(x, next_by, accuracy)
        for label, group in new_x:
            _grow_tree(group, deepcopy(by), target, accuracy, path + [label],
                       paths_out, values_out)
    return paths_out, values_out


def slice_dataframe(x, by, accuracy):
    """
    Slice a DataFrame *x* into a group of equally sized dataframes based on
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
