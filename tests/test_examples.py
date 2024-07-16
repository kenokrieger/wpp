"""
Confirm that the examples are not throwing any errors.
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
import matplotlib.pyplot as plt

from zephyros.examples import get_sample_data, plot_prediction


def test_physical_predictor_example():
    from zephyros.physical_predictor import predict
    # use 500 example values
    x = get_sample_data().iloc[10_000:10_500]
    y, uy = predict(x)
    fig, ax = plot_prediction(x.index, (y, y - uy, y + uy), x["power_measured"],
                              title="Power Prediction based on Physical Calculations",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_empirical_predictor_example():
    from zephyros.empirical_predictor import (learn_and_predict, PREDICT_KEY,
                                              UNCERTAINTY_KEY)
    x = get_sample_data()
    nrows = x.shape[0]
    # use 99 % of data for "learning"
    learn_predict_split = int(0.99 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y = learn_and_predict(learn_data, predict_data,
                          features, target, accuracy=12)
    plot_values = (y, y[PREDICT_KEY] - y[UNCERTAINTY_KEY],
                   y[PREDICT_KEY] + y[UNCERTAINTY_KEY])
    fig, ax = plot_prediction(y.index, plot_values, predict_data["power_measured"],
                              title="Power Prediction based on Historically Measured Values",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_svm_predictor_example():
    from zephyros import svm_predictor
    x = get_sample_data()
    nrows = x.shape[0]
    # use 99 % of data for learning
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y = svm_predictor.learn_and_predict(learn_data, predict_data,
                                        features, target)
    fig, ax = plot_prediction(predict_data.index, (y, ),
                              predict_data["power_measured"],
                              title="Power Prediction using SVR",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_rvm_predictor_example():
    from zephyros.rvm_predictor import learn_and_predict
    x = get_sample_data().iloc[:2_000]
    nrows = x.shape[0]
    # use 99 % of data for learning
    learn_predict_split = int(0.95 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y, std_y = learn_and_predict(learn_data, predict_data, features, target)

    fig, ax = plot_prediction(predict_data.index,
                              (y, y - std_y, y + std_y),
                              predict_data["power_measured"],
                              title="Power Prediction using RVM",
                              xlabel="Time index", ylabel="Power in kW")

    assert True


def test_boost_predictor_example():
    from zephyros.boost_predictor import learn_and_predict
    x = get_sample_data()
    nrows = x.shape[0]
    # use 99 % of data for learning
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    # predict with 1 cross validation
    y, ly, uy = learn_and_predict(learn_data, predict_data, features, target,
                                  xvalidate=1)
    # average the results of each cross validation
    y = y.mean(axis=0)
    ly = ly.mean(axis=0)
    uy = uy.mean(axis=0)
    # visualise the results
    fig, ax = plot_prediction(predict_data.index, (y, ly, uy),
                              predict_data["power_measured"],
                              title="Power Prediction using XGBoost",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def ann_predictor_example():
    from zephyros.ann_predictor import learn_and_predict
    x = get_sample_data()
    nrows = x.shape[0]
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y = learn_and_predict(learn_data, predict_data, features, target)
    y = y.mean(axis=0)
    # visualise the results
    fig, ax = plot_prediction(predict_data.index, (y, ),
                              predict_data["power_measured"],
                              title="Power Prediction with ANN",
                              xlabel="Time index", ylabel="Power in kW")
    assert True
