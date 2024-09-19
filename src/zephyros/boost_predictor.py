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
import xgboost

from zephyros._utils import sample_and_scale


def learn_and_predict(learn_data, predict_data, features, target,
                      test_percentage=0.33, xvalidate=0, seed=None,
                      compute_margins=False, xgboost_options=None):
    """
    Convenience function that combines boost_predictor.learn and
    boost_predictor.predict. Given learn and predict data, learn a model with
    the first and use it to predict the latter. Return the predicted values
    and the lower and upper bound of their 90% intervals.

    Args:
        learn_data (pandas.DataFrame): The data to use for learning the model.
        predict_data (pandas.DataFrame): The data to use for the prediction.
        features (list): The features to use for learning and predicting.
        target (list): The target(s) to predict.
        test_percentage (float): Percentage of *learn_data* to use for testing.
        xvalidate (int): Choose the number of cross validations.
            Defaults to 0.
        seed (int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        compute_margins (bool): Enable computing of uncertainty margins for
            the predictions. Defaults to False.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        tuple or numpy.ndarray: The predicted values and, optionally, the lower
         and upper bound of their 90% confidence intervals.

    """
    if compute_margins:
        return _learn_and_predict_with_margins(
            learn_data, predict_data, features, target, test_percentage,
            xvalidate, seed, xgboost_options)

    return _learn_and_predict(learn_data, predict_data, features, target,
                              test_percentage, xvalidate, seed, xgboost_options)


def _learn_and_predict_with_margins(learn_data, predict_data, features, target,
                                    test_percentage, xvalidate, seed,
                                    xgboost_options):
    """
    Given learn and predict data, learn a model with the first and use it to
    predict the latter. Return the predicted values and the lower and upper
    bound of their 90% intervals.

    Args:
        learn_data(pandas.DataFrame): The data to use for learning the model.
        predict_data(pandas.DataFrame): The data to use for the prediction.
        features(list): The features to use for learning and predicting.
        target(list): The target(s) to predict.
        test_percentage(float): Percentage of *learn_data* to use for testing.
        xvalidate(int): Choose the number of cross validations.
            Defaults to 0.
        seed(int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        xgboost_options(dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        tuple: The predicted values and the lower and upper bound of their 90%
            confidence intervals.

    """
    random_state = np.random.default_rng(seed=seed)
    nrows = predict_data.shape[0]
    predicted = np.empty((xvalidate + 1, nrows))
    lower_bound = np.empty((xvalidate + 1, nrows))
    upper_bound = np.empty((xvalidate + 1, nrows))
    for i in range(xvalidate + 1):
        models = single_learn(learn_data, features, target, test_percentage,
                              random_state, xgboost_options)
        model, lower_bound_model, upper_bound_model = models
        predicted[i] = predict(*model, predict_data[features].to_numpy())
        lower_bound[i] = predict(*lower_bound_model, predict_data[features].to_numpy())
        upper_bound[i] = predict(*upper_bound_model, predict_data[features].to_numpy())
    return lower_bound, predicted, upper_bound


def _learn_and_predict(learn_data, predict_data, features, target,
                       test_percentage, xvalidate, seed, xgboost_options):
    """
    Given learn and predict data, learn a model with the first and use it to
    predict the latter. Return the predicted values.

    Args:
        learn_data (pandas.DataFrame): The data to use for learning the model.
        predict_data (pandas.DataFrame): The data to use for the prediction.
        features (list): The features to use for learning and predicting.
        target (list): The target(s) to predict.
        test_percentage (float): Percentage of *learn_data* to use for testing.
        xvalidate (int): Choose the number of cross validations.
            Defaults to 0.
        seed (int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        numpy.ndarray: The predicted values.

    """
    random_state = np.random.default_rng(seed=seed)
    nrows = predict_data.shape[0]
    predicted = np.empty((xvalidate + 1, nrows))
    for i in range(xvalidate + 1):
        model = learn(learn_data, features, target, test_percentage,
                      random_state, xgboost_options)
        predicted[i] = predict(*model, predict_data[features].to_numpy())
    return predicted


def single_learn(learn_data, features, target, test_percentage, random_state,
                 xgboost_options):
    """
    Given learn data, features and a target, learn a model to fit the data
    as well as two models for the lower 0.05 percentile and the 0.95 percentile
    respectively.

    Args:
        learn_data (pandas.DataFrame): The data for learning the models.
        features (list): The features to use in the learning process.
        target (list): The target(s) for the learning process.
        test_percentage (float): Percentage of *learn_data* to use for testing.
        random_state (np.random.Generator): Random generator for the sampling.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        tuple: The models to fit the data and the 0.05 and 0.95 percentiles
            respectively.

    """
    options = dict()
    if xgboost_options is not None:
        options.update(xgboost_options)
    model = learn(learn_data, features, target, test_percentage, random_state,
                  xgboost_options)
    options.update(dict(objective="reg:quantileerror", quantile_alpha=0.05))
    lower_bound_model = learn(learn_data, features, target, test_percentage,
                              random_state, xgboost_options=options)
    options.update(dict(quantile_alpha=0.95))
    upper_bound_model = learn(learn_data, features, target, test_percentage,
                              random_state, xgboost_options=options)
    return model, lower_bound_model, upper_bound_model


def learn(x, features, target, test_percentage=0.33,
          random_state=1, xgboost_options=None, scale=True):
    """

    Args:
        x (pandas.DataFrame): The data to use for learning a model.
        features (list): The features to use in the learning process.
        target (list): The target(s) for the learning process.
        test_percentage (float): Percentage of *x* to use for testing.
        random_state (np.random.Generator): Random generator for the sampling.
        xgboost_options (dict): Options to pass to xgboost.XGBRegressor.
        scale (bool): Scale the feature and target values. Defaults to True.

    Returns:
        xgboost.XGBRegressor: The learned model.

    """
    options = dict(booster='gbtree', objective='reg:squarederror')
    if xgboost_options is not None:
        options.update(xgboost_options)

    scaler, values = sample_and_scale(x, features, target, test_percentage,
                                      random_state,
                                      method="standard" if scale else None)
    reg = xgboost.XGBRegressor(**options)
    reg.fit(values[0], values[1], eval_set=[values[2:]], verbose=100)
    return reg, scaler


def predict(model, scaler, x):
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(xgb.XGBRegressor): The learned model.
        scaler(tuple): The feature and target scaler that were used in
            the learning of the model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.array: The predicted values for the target learned.

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

