"""
Module containing utility functions for scaling and sampling data.
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale(x: np.ndarray, y: np.ndarray, method:str|None="standard") -> tuple:
    """
    Scale the data for the learning process and return the scaled
    data and the scalers used in the process.

    Args:
        x (np.ndarray): The unscaled learning data containing the feature
            values.
        y (np.ndarray): The unscaled learning data containing the target values.
        method (str): The type of scaler to use, e.g. standard or minmax.
            Defaults to 'standard'.

    Returns:
        tuple: The scalers and the scaled values.

    """
    scaler_mapping = {"standard": StandardScaler, "minmax": MinMaxScaler}
    try:
        scaler = scaler_mapping[method]
    except KeyError:
        raise KeyError(f"Method {method} not supported. Supported methods are: "
                       ", ".join(scaler_mapping.keys()))
    feature_scaler = scaler()
    target_scaler = scaler()
    feature_scaler.fit(x)
    x_scaled = feature_scaler.transform(x)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    target_scaler.fit(y)
    y_scaled = target_scaler.transform(y)
    return (feature_scaler, target_scaler), (x_scaled, y_scaled)


def sample_and_scale(x: pd.DataFrame, features: list, target: list, test_percentage: float,
                     random_state: np.random.Generator|int|None,
                     method:str|None="standard") -> tuple:
    """
    Sample and scale the data for the learning process and return the scaled
    data and the scalers used in the process.

    Args:
        x (pandas.DataFrame): The unsampled learning data.
        features (list): Column names of the features to use in the learning
            process.
        target (list): Column name of the target for the learning process.
        test_percentage (float): Percentage of data to use for testing.
        random_state (np.random.Generator or None): The random state to use for
            the sampling.
        method (str): The method to use for the scaling, e.g. standard or
            minmax. If method is None, no scaling is applied. Defaults to
            'standard'.

    Returns:
        tuple: The scalers and the scaled values.

    """
    test = x.sample(frac=test_percentage, random_state=random_state)
    # complement of the sampled data
    train = x.loc[x.index.difference(test.index)]
    x_train = train[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    if method is None:
        return None, (x_train, y_train, x_test, y_test)

    scalers, scaled_values = scale(x_train, y_train, method)
    feature_scaler, target_scaler = scalers
    x_scaled, y_scaled = scaled_values
    x_test_scaled = feature_scaler.transform(x_test)
    y_test_scaled = target_scaler.transform(y_test)
    return scalers, (x_scaled, y_scaled, x_test_scaled, y_test_scaled)
