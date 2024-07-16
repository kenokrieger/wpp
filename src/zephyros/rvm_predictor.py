"""
Module for using relevance vector machine to predict the estimated power
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
from sklearn_rvm import EMRVR
from zephyros._utils import scale


def learn_and_predict(learn_data, predict_data, features, target,
                      rvm_options=None):
    """
    Given learn data and predict data, learn a relevance vector machine and use
    it on the predict data. Return the predicted values.

    Args:
        learn_data(pandas.DataFrame): Data to use for learning the machine.
        predict_data(pandas.DataFrame: Data to use for the prediction.
        features(list): The features to use for the learning and predicting.
        target(list): The target of the learning and predicting.
        rvm_options(dict or None): Options for the EMRVR regressor. Defaults to
            None.

    Returns:
        numpy.ndarray: The predicted values and their standard deviation.

    """
    x_in = learn_data[features].to_numpy()
    y_in = learn_data[target].to_numpy().ravel()
    x_pred = predict_data[features].to_numpy()
    model, scaler = learn(x_in, y_in, rvm_options)
    return predict(model, scaler, x_pred)


def learn(x_in, y_in, rvm_options=None):
    """
    Learn a relevance vector machine given feature values *x_in* and
    target values *y_in*. Return the learned model.

    Args:
        x_in(np.ndarray): Feature values to use for learning the model.
        y_in(np.ndarray: Target value for the learning process.
        rvm_options(dict or None): Options for the EMRVR regressor. Defaults to
            None.

    Returns:
        tuple: The learned model and the scaler.

    """
    rvm_config = dict(kernel="rbf", gamma="auto")
    if rvm_options is not None:
        rvm_config.update(rvm_options)
    model = EMRVR(**rvm_config)
    scaler, values = scale(x_in, y_in)
    model.fit(values[0], values[1].ravel())
    return model, scaler


def predict(model, scaler, x):
    """
    Predict values given a learned model and features *x_pred*.
    Args:
        model(sklearn_rvm.EMRVR): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(numpy.ndarray): Feature values to use for the prediction.

    Returns:
        np.ndarray: The predicted values and their standard deviation.
    """
    x_scaled = scaler[0].transform(x)
    y, std_y = model.predict(x_scaled, return_std=True)
    real_y = scaler[1].inverse_transform(y.reshape(-1, 1)).ravel()
    real_std_y = scaler[1].inverse_transform(std_y.reshape(-1, 1)).ravel()
    return real_y, real_std_y
