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


def learn_and_predict(learn_data, predict_data, features, target,
                      test_percentage=0.33, xvalidate=None, seed=None,
                      xgboost_options=None):
    """
    Convenience function that combines boost_predictor.learn and
    boost_predictor.predict. Given learn and predict data, learn a model with
    the first and use it to predict the latter. Return the predicted values
    and the lower and upper bound of their 90% intervals.

    Args:
        learn_data(pandas.DataFrame): The data to use for learning the model.
        predict_data(pandas.DataFrame): The data to use for the prediction.
        features(list): The features to use for learning and predicting.
        target(list): The target(s) to predict.
        test_percentage(float): Percentage of *learn_data* to use for testing.
        xvalidate(int or None): Choose the number of cross validations.
            Defaults to None.
        seed(int or None): Set a seed for the sampling of test data for
            reproducibility. Defaults to None.
        xgboost_options(dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        tuple: The predicted values, the lower and upper bound of their 90%
            intervals.

    """
    random_state = np.random.default_rng(seed=seed)
    if not xvalidate:
        models = single_learn(learn_data, features, target, test_percentage,
                              random_state, xgboost_options)
        model, lower_bound_model, upper_bound_model = models
        return (predict(model, predict_data[features]),
                predict(lower_bound_model, predict_data[features]),
                predict(upper_bound_model, predict_data[features]))

    nrows = predict_data.shape[0]
    predicted  = np.empty((xvalidate, nrows))
    lower_bound = np.empty((xvalidate, nrows))
    upper_bound = np.empty((xvalidate, nrows))

    for i in range(xvalidate):
        models = single_learn(learn_data, features, target, test_percentage,
                              random_state, xgboost_options)
        model, lower_bound_model, upper_bound_model = models
        predicted[i] = predict(model, predict_data[features])
        lower_bound[i] = predict(lower_bound_model, predict_data[features])
        upper_bound [i] = predict(upper_bound_model, predict_data[features])
    return predicted, lower_bound, upper_bound


def single_learn(learn_data, features, target, test_percentage, random_state,
                 xgboost_options):
    """
    Given learn data, features and a target, learn a model to fit the data
    as well as two models for the lower 0.05 percentile and the 0.95 percentile
    respectively.

    Args:
        learn_data (pandas.DataFrame): The data for learning the models.
        features(list): The features to use in the learning process.
        target(list): The target(s) for the learning process.
        test_percentage(float): Percentage of *learn_data* to use for testing.
        random_state(np.random.Generator): Random generator for the sampling.
        xgboost_options(dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        tuple: The models to fit the data and the 0.05 and 0.95 percentiles
            respectively.

    """
    model = learn(learn_data, features, target, test_percentage, random_state, xgboost_options)
    lower_bound_model = learn(learn_data, features, target, test_percentage, random_state,
                              xgboost_options=dict(objective="reg:quantileerror", quantile_alpha=0.05))
    upper_bound_model = learn(learn_data, features, target, test_percentage, random_state,
                              xgboost_options=dict(objective="reg:quantileerror", quantile_alpha=0.95))
    return model, lower_bound_model, upper_bound_model


def learn(x, features, target, test_percentage=0.33,
          random_state=1, xgboost_options=None):
    """

    Args:
        x(pandas.DataFrame): The data to use for learning a model.
        features(list): The features to use in the learning process.
        target(list): The target(s) for the learning process.
        test_percentage(float): Percentage of *x* to use for testing.
        random_state(np.random.Generator): Random generator for the sampling.
        xgboost_options(dict): Options to pass to xgboost.XGBRegressor.

    Returns:
        xgboost.XGBRegressor: The learned model.

    """
    options = dict(base_score=1.5e3, booster='gbtree', n_estimators=10_000,
                   device="cuda", subsample=0.8,
                   early_stopping_rounds=500, objective='reg:squarederror',
                   max_depth=24, learning_rate=0.001)
    if xgboost_options is not None:
        options.update(xgboost_options)

    test = x.sample(frac=test_percentage, random_state=random_state)
    # complement of the sampled data
    train = x.iloc[x.index.difference(test.index)]

    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    reg = xgboost.XGBRegressor(**options)
    reg.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            verbose=100)
    return reg


def predict(model, x):
    """
    Given feature values *x* and a learned *model*, predict values for the
    target from the learning process of the model.

    Args:
        model(xgboost.XGBRegressor): The learned model.
        x(pandas.DataFrame): Feature values.

    Returns:
        np.array: The predicted values for the target learned.

    """
    return model.predict(x)
