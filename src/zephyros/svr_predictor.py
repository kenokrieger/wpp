"""
Module for using support vector machine to predict the estimated power
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

from sklearn.svm import SVR, LinearSVR

from zephyros._utils import scale


def learn_and_predict(learn_data: pd.DataFrame, predict_data: pd.DataFrame,
                      features: list, target:list, svr_options: dict|None=None) -> np.ndarray:
    """
    Given learn data and predict data, learn a support vector machine and use
    it on the predict data. Return the predicted values.

    Args:
        learn_data(pandas.DataFrame): Data to use for learning the machine.
        predict_data(pandas.DataFrame: Data to use for the prediction.
        features(list): The features to use for the learning and predicting.
        target(list): The target of the learning and predicting.
        svr_options (dict): Options for the Regressor.
        bagging_options (dict): Options for the Bagging Regressor

    Returns:
        numpy.ndarray: The predicted values.

    """
    x_in = learn_data[features].to_numpy()
    y_in = learn_data[target].to_numpy().ravel()
    x_pred = predict_data[features].to_numpy()
    model, scaler = learn(x_in, y_in, svr_options)
    return predict(model, scaler, x_pred)


def learn(x_in: np.ndarray, y_in: np.ndarray, svr_options: dict|None=None) -> tuple:
    """
    Learn a support vector machine given feature values *x_in* and
    target values *y_in*. If svr_options['kernel'] == 'linear' this method uses
    the optimized LinearSVR model else it uses the SVR model. Return the learned
    model.

    Args:
        x_in (np.ndarray): Feature values to use for learning the model.
        y_in (np.ndarray): Target value for the learning process.
        svr_options (dict): Options for the Regressor.
        bagging_options (dict): Options for the Bagging Regressor

    Returns:
        tuple: The learned model and the scaler.

    """
    svr_config = dict(kernel="rbf")
    if svr_options is not None:
        svr_config.update(svr_options)
    if svr_config["kernel"] == "fast-linear":
        regressor = LinearSVR
        del svr_config["kernel"]
    else:
        regressor = SVR

    scaler, values = scale(x_in, y_in)
    model = regressor(**svr_config)
    model.fit(values[0], values[1].ravel())
    return model, scaler


def predict(model: SVR, scaler: tuple, x: np.ndarray) -> np.ndarray:
    """
    Predict values given a learned model and features *x_pred*.
    Args:
        model(sklearn.svm.SVR): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(numpy.ndarray): Feature values to use for the prediction.

    Returns:
        np.ndarray: The predicted values.
    """
    x_scaled = scaler[0].transform(x)
    y = model.predict(x_scaled).reshape(-1, 1)
    return scaler[1].inverse_transform(y).ravel()
