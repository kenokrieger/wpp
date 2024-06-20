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
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor


def learn_and_predict(learn_data, predict_data, features, target,
                      svr_options=None):
    """
    Given learn data and predict data, learn a support vector machine and use
    it on the predict data. Return the predicted values.

    Args:
        learn_data(pandas.DataFrame): Data to use for learning the machine.
        predict_data(pandas.DataFrame: Data to use for the prediction.
        features(list): The features to use for the learning and predicting.
        target(list): The target of the learning and predicting.
        svr_options (dict): Options for the BaggingRegressor.

    Returns:
        numpy.ndarray: The predicted values.

    """
    x_in = learn_data[features].to_numpy()
    y_in = learn_data[target].to_numpy().ravel()
    x_pred = predict_data[features].to_numpy()
    model = learn(x_in, y_in, svr_options)
    return predict(model, x_pred)


def predict(model, x_pred):
    """
    Predict values given a learned model and features *x_pred*.
    Args:
        model(sklearn.svm.SVR): The learned model.
        x_pred(numpy.ndarray): Feature values to use for the prediction.

    Returns:
        np.ndarray: The predicted values.
    """
    return model.predict(x_pred)


def learn(x_in, y_in, svr_options=None):
    """
    Learn a support vector machine given feature values *x_in* and
    target values *y_in*. Return the learned model.

    Args:
        x_in(np.ndarray): Feature values to use for learning the model.
        y_in(np.ndarray: Target value for the learning process.
        svr_options (dict): Options for the BaggingRegressor.

    Returns:
        sklearn.svm.SVR: The learned model.
    """
    options = dict(bootstrap=True, n_estimators=12, n_jobs=-1, max_samples=0.66)
    if svr_options is not None:
        options.update(svr_options)
    model = BaggingRegressor(SVR(kernel="rbf", gamma="auto"), **options)
    model.fit(x_in, y_in)
    return model
