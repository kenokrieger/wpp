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


def test_physical_predictor_example():
    from zephyros.sample_data import get_sample_data
    from zephyros.physical_predictor import predict
    from zephyros.examples import plot_prediction
    # use 500 example values
    x = get_sample_data().iloc[10_000:10_500]
    y, uy = predict(x)
    fig, ax = plot_prediction(x.index, (y, y - uy, y + uy), x["power_measured"],
                              title="Power Prediction based on Physical Calculations",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_svr_predictor_example():
    from zephyros.sample_data import get_sample_data
    from zephyros.svr_predictor import learn_and_predict
    from zephyros.examples import plot_prediction
    x = get_sample_data()
    nrows = x.shape[0]
    # use 99 % of data for learning
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y = learn_and_predict(learn_data, predict_data,
                          features, target)
    fig, ax = plot_prediction(predict_data.index, (y, ),
                              predict_data["power_measured"],
                              title="Power Prediction using SVR",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_rvr_predictor_example():
    from zephyros.sample_data import get_sample_data
    from zephyros.rvr_predictor import learn_and_predict
    from zephyros.examples import plot_prediction
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
                              (y, y - 2 * std_y, y + 2 * std_y),
                              predict_data["power_measured"],
                              title="Power Prediction using RVM",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def test_xgb_predictor_example():
    from zephyros.sample_data import get_sample_data
    from zephyros.xgb_predictor import learn_and_predict
    from zephyros.examples import plot_prediction
    x = get_sample_data()
    nrows = x.shape[0]
    # use 99 % of data for learning
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]
    y = learn_and_predict(learn_data, predict_data, features, target).ravel()
    uy = learn_and_predict(learn_data, predict_data, features, target,
                           xgboost_options={"objective": "reg:quantileerror", "quantile_alpha": 0.95}).ravel()
    ly = learn_and_predict(learn_data, predict_data, features, target,
                           xgboost_options={"objective": "reg:quantileerror", "quantile_alpha": 0.05}).ravel()
    # visualise the results
    fig, ax = plot_prediction(predict_data.index, (y, ly, uy),
                              predict_data["power_measured"],
                              title="Power Prediction using XGBoost",
                              xlabel="Time index", ylabel="Power in kW")
    assert True


def ann_predictor_example():
    import numpy as np
    from zephyros.sample_data import get_sample_data
    from zephyros.ann_predictor import learn_and_predict
    from zephyros.examples import plot_prediction
    x = get_sample_data()
    nrows = x.shape[0]
    learn_predict_split = int(0.999 * nrows)
    learn_data = x.iloc[:learn_predict_split]
    predict_data = x.iloc[learn_predict_split:]
    features = ["wind_speed", "temperature"]
    target = ["power_measured"]

    config = {
        # layers of the model
        "layers": [
            {"units": 50, "kernel_initializer": "normal", "activation": "sigmoid"},
            {"units": 25, "kernel_initializer": "normal", "activation": "sigmoid"},
            {"units": 2, "kernel_initializer": "normal"},
        ],
        # options for compiling the model
        "compile": {"loss": "loglikelihood", "optimizer": "adam"},
        # callbacks and their respective options
        "callbacks": {"EarlyStopping": {"monitor": "val_loss", "patience": 5}},
        # options for the learning process
        "options": {"batch_size": 200, "epochs": 1_000, "verbose": 1}
    }

    y = learn_and_predict(learn_data, predict_data, features, target, config=config)
    # visualise the results
    fig, ax = plot_prediction(predict_data.index,
                              (y[:, 0], y[: , 0] - 2 * np.sqrt(y[:, 1]), y[:, 0] + 2 * np.sqrt(y[:, 1])),
                              predict_data["power_measured"],
                              title="Power Prediction with ANN",
                              xlabel="Time index", ylabel="Power in kW")
    assert True
