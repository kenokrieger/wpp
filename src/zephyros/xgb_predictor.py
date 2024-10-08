"""
Module for using extreme gradient boosting to predict the estimated power
output of a wind turbine.
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
import xgboost

from zephyros._utils import sample_and_scale


def learn_and_predict(learn_data: pd.DataFrame, predict_data: pd.DataFrame,
                      features: list, target: list, validate_percentage: float=0.33,
                      seed: int|None=None, xgboost_options:dict|None=None) -> np.ndarray:
    """
    Given learn and predict data, learn a model with the first and use it to
    predict the latter. Return the predicted values.

    Args:
        learn_data (pandas.DataFrame): The data to use for learning the model.
        predict_data (pandas.DataFrame): The data to use for the prediction.
        features (list): The features to use for learning and predicting.
        target (list): The target(s) to predict.
        validate_percentage (float): Percentage of *learn_data* to use for validation.
        seed (int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        numpy.ndarray: The predicted values.

    """
    model = learn(learn_data, features, target, validate_percentage,
                  seed, xgboost_options)
    return predict(*model, predict_data[features].to_numpy())


def learn(x: pd.DataFrame, features: list, target: list,
          validate_percentage: float=0.33, random_state: int|None=None,
          xgboost_options: dict|None=None, scale: bool=True) -> tuple:
    """

    Args:
        x (pandas.DataFrame): The data to use for learning a model.
        features (list): The features to use in the learning process.
        target (list): The target(s) for the learning process.
        validate_percentage (float): Percentage of *learn_data* to use for validation.
        random_state (np.random.Generator): Random generator for the sampling.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.
        scale (bool): Scale the feature and target values. Defaults to True.

    Returns:
        tuple: The learned model and the feature and target scaler.

    """
    options = dict(booster="gbtree", objective="reg:squarederror")
    if xgboost_options is not None:
        options.update(xgboost_options)

    scaler, values = sample_and_scale(x, features, target, validate_percentage,
                                      random_state, method="standard" if scale else None)
    reg = xgboost.XGBRegressor(**options)
    reg.fit(values[0], values[1], eval_set=[values[2:]], verbose=100)
    return reg, scaler


def predict(model: xgboost.XGBRegressor, scaler: tuple, x: np.ndarray) -> np.ndarray:
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(xgb.XGBRegressor): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.ndarray: The predicted values for the target learned.

    """
    if scaler is not None:
        x_scaled = scaler[0].transform(x)
        y = model.predict(x_scaled)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        prediction = scaler[1].inverse_transform(y)
    else:
        prediction = model.predict(x)
    if prediction.shape[1] == 1:
        return prediction.ravel()
    return prediction
