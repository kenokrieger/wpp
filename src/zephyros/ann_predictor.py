"""
Module for using artificial neural networks to predict the estimated power
output of a wind turbine.
"""
import keras
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
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed
from keras import ops
from zephyros._utils import sample_and_scale


def learn_and_predict(learn_data: pd.DataFrame, predict_data: pd.DataFrame,
                      features: list, target: list,
                      validate_percentage: float=0.33, seed: int|None=None,
                      scale_method: str|None="standard", config: dict|None=None) -> np.ndarray:
    """
    Convenience function combining ann_predictor.learn and ann_predictor.predict
    into a single function. Given feature values and a target to predict, learn
    a model and use it to predict the target variable in the *predict_data*.

    Args:
        learn_data(pandas.DataFrame): The data to use for learning the model.
        predict_data(pandas.DataFrame): The data to use for the prediction.
        features(list): The features to use for learning and predicting.
        target(list): The target(s) to predict.
        validate_percentage(float): Percentage of *learn_data* to use for testing.
        seed(int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        scale_method (str): The method to use for the feature and target
            scaling, e.g. standard or minmax. If scale_method is None, no
            scaling is applied. Defaults to 'standard'.
        config (dict): A configuration for the structure of the neural network
            and the learning process.

    Returns:
        np.ndarray: The predicted values.

    """
    if seed is not None:
        set_random_seed(seed)
    random_state = np.random.default_rng(seed=seed)
    model, scaler = learn(learn_data, features, target, validate_percentage,
                          random_state, scale_method, config)
    x_pred = predict_data[features].to_numpy(dtype=float)
    return predict(model, scaler, x_pred)


def learn(x: pd.DataFrame, features: list, target: list, validation_percentage: float=0.33,
          random_state: np.random.Generator|None=None,
          scale_method: str|None="standard", config: dict|None=None) -> tuple:
    """
    Args:
        x(pandas.DataFrame): The data to use for learning a model.
        features(list): The features to use in the learning process.
        target(list): The target(s) for the learning process.
        validation_percentage(float): Percentage of *x* to use for validation.
        random_state(np.random.Generator or None): Random generator for the
            sampling.
        scale (bool): Scale the in- and output values. Defaults to True.
        scale_method (str): The method to use for the feature and target
            scaling, e.g. standard or minmax. If scale_method is None, no
            scaling is applied. Defaults to 'standard'.
        config (dict or None): A configuration for the structure of the neural
            network and the learning process. Defaults to None.

    Returns:
        tuple: The learned model and the scalers.

    """
    cb_map = {"EarlyStopping": EarlyStopping}
    default_config = {
        "layers": [
            {"units": 50, "kernel_initializer": "normal", "activation": "relu"},
            {"units": 3, "kernel_initializer": "normal", "activation": "tanh"},
            {"units": 1, "kernel_initializer": "normal"},
        ],
        "compile": {"loss": "mean_squared_error", "optimizer": "adam"},
        "callbacks": {"EarlyStopping": {"monitor": "val_loss", "patience": 5}},
        "options": {"batch_size": 200, "epochs": 1_000, "verbose": 1}
    }
    if config is not None:
        default_config.update(config)
    config = default_config
    _set_loss_function(config)

    scaler, values = sample_and_scale(x, features, target, validation_percentage,
                                      random_state, method=scale_method)

    model = Sequential([Dense(**c) for c in config["layers"]])
    model.compile(**config["compile"])

    cb_opt = config["callbacks"]
    callbacks = [cb_map[cb](**opt) for cb, opt in cb_opt.items() if cb in cb_map]
    model.fit(values[0], values[1],
              validation_data=values[2:],
              callbacks=callbacks,
              **config["options"])
    return model, scaler


def predict(model: keras.Sequential, scaler: tuple, x: pd.DataFrame) -> np.ndarray:
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(keras.models.Sequential): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.array: The predicted values for the target learned.

    """
    used_nllh = "log_likelihood" in model.loss.__name__
    print(model.loss.__name__)
    # x_scaled = scaler[0].transform(x)
    # y = model.predict(x_scaled)

    # prediction = scaler[1].inverse_transform(y)
    # lower = prediction[:, 0] - 2 * np.sqrt(prediction[:, 1])
    # upper = prediction[:, 0] + 2 * np.sqrt(prediction[:, 1])


    if scaler is not None:
        x_scaled = scaler[0].transform(x)
    else:
        x_scaled = x
    y = model.predict(x_scaled)

    if used_nllh:
        y[:, 1] = np.exp(y[:, 1])

    if scaler is not None:
        prediction = scaler[1].inverse_transform(y)
    else:
        prediction = y
    if prediction.shape[1] == 1:
        return prediction.ravel()
    return prediction


def _set_loss_function(config: dict) -> None:
    """
    Replace string choices for loss functions to their functional implementations.

    Args:
        config (dict): The configuration of the ANN.

    Returns:
        None.

    """
    if config["compile"]["loss"] == "quantile" and "quantile_alpha" not in config["compile"]:
        raise KeyError("Quantile loss was chosen but 'quantile_alpha' was not specified.")

    if config["compile"]["loss"] == "quantile":
        qa = config["compile"]["quantile_alpha"]

        def quantile_loss(y_true, y_pred):
            errors = y_true - y_pred
            combined = ops.append((qa - 1) * errors, qa * errors, axis=1)
            loss = ops.max(combined, axis=0)
            return ops.mean(ops.abs(loss))

        config["compile"]["loss"] = quantile_loss
        del config["compile"]["quantile_alpha"]

    elif config["compile"]["loss"] == "loglikelihood":
        def log_likelihood(y_true, y_pred):
            return ops.mean(ops.square(y_pred[:, 0] - y_true[:, 0]) / ops.exp(
                y_pred[:, 1]) + y_pred[:, 1])
        config["compile"]["loss"] = log_likelihood
