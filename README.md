# <img src="https://raw.githubusercontent.com/kenokrieger/zephyros/master/LOGO.png" alt="zephyros" height=60>

zephyros, named after one of the Greek gods of wind,
is a package designed for predicting power output from wind turbines
(or wind farms) using different methods ranging from physical power
calculations to machine learning.

## Install

You can install the package directly from [GitHub](https://github.com/kenokrieger/zephyros) by running

```bash
pip install git+https://github.com/kenokrieger/zephyros
```

or download the source code of the latest
[release](https://github.com/kenokrieger/zephyros/releases/latest)
and then run

```bash
python3 -m pip install .
```

in the top-level directory to install the package.

## Usage

### About the modules

All the machine learning modules
(`zephyros.svm_predictor`, `zephyros.rvm_predictor`,
`zephyros.boost_predictor`, `zephyros.ann_predictor`)
are wrappers for existing libraries. Their functions are to be understood
as convenience functions (see [Examples](%22#Examples%22) below). In the
following the customisation options for each module will be explained.

#### zephyros.svm_predictor

By default the `svm_predictor` module use the
[sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)
regressor. Options can be passed to the regressor by using the `svr_options`
parameter. For the option `kernel='fast-linear'` the
[sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)
regressor will be used instead. Additionally, if desired, it can use the
[sklearn.ensemble.BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
using the `bagging_options` parameter with `n_estimators > 1`.

#### zephyros.rvm_predictor

The `rvm_predictor` uses the
[sklearn_rvm.em_rvm.EMRVR](https://sklearn-rvm.readthedocs.io/en/latest/generated/sklearn_rvm.em_rvm.EMRVR.html)
regressor. It can be configured using the `rvm_options` parameter.

#### zephyros.boost_predictor

The `boost_predictor` uses the
[xgboost.XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn).
It can be configured using the `xgboost_options` parameter.

#### zephyros.ann_predictor

The `ann_predictor` is a wrapper for `keras`. Here, not only options can
be passed but also a configuration of dense layers. For example

```python
config = {
    # layers of the model
    "layers": [
        {"units": 50, "kernel_initializer": "normal", "activation": "relu"},
        {"units": 20, "kernel_initializer": "normal", "activation": "relu"},
        {"units": 10, "kernel_initializer": "normal", "activation": "relu"},
        {"units": 3, "kernel_initializer": "normal", "activation": "tanh"},
        {"units": 1, "kernel_initializer": "normal"},
    ],
    # options for compiling the model
    "compile": {"loss": "mean_squared_error", "optimizer": "adam"},
    # callbacks and their respective options
    "callbacks": {"EarlyStopping": {"monitor": "val_loss", "patience": 5}},
    # options for the learning process
    "options": {"batch_size": 200, "epochs": 1_000, "verbose": 1}
}
```

### Examples

#### Example 1: Use the physical prediction method

```python
from zephyros.examples import get_sample_data, plot_prediction
from zephyros.physical_predictor import predict

# use 500 example values
x = get_sample_data().iloc[10_000:10_500]
y, uy = predict(x)
fig, ax = plot_prediction(x.index, (y, y - uy, y + uy), x["power_measured"],
                          title="Power Prediction based on Physical Calculations",
                          xlabel="Time index", ylabel="Power in kW")
plt.show()
```

#### Example 2: Use the empirical prediction method

```python
from zephyros.examples import get_sample_data, plot_prediction
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
plt.show()
```

#### Example 3: Use Support Vector Regression

```python
from zephyros.examples import get_sample_data, plot_prediction
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
plt.show()
```

#### Example 4: Use Relevance Vector Regression

```python
from zephyros.examples import get_sample_data, plot_prediction
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


plt.show()
```

#### Example 5: Use Extreme Gradient Boosting (xgboost package)

```python
from zephyros.examples import get_sample_data, plot_prediction
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
plt.show()
```

#### Example 6: Use Artificial Neural Networks

```python
from zephyros.examples import get_sample_data, plot_prediction
from zephyros.ann_predictor import learn_and_predict
x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y = learn_and_predict(learn_data, predict_data, features, target)
# learn and predict returns a column vector
y = y.mean(axis=0)
# visualise the results
fig, ax = plot_prediction(predict_data.index, (y, ),
                          predict_data["power_measured"],
                          title="Power Prediction with ANN",
                          xlabel="Time index", ylabel="Power in kW")
plt.show()
```

## Development

### Tests

**IMPORTANT NOTICE: Test data in the test_data directory is
licensed under CC BY Attribution 4.0 International license**

Tests can be run by executing

```bash
python3 -m pytest
```

in the highest
directory.

Tests are only implemented for modules that contain large self implemented
methods. Modules that only serve as API for well-tested packages
(e.g. `zephyros.boost_predictor`) are not tested.

## License

This project is licensed under GNU GENERAL PUBLIC LICENSE.
For more details see the LICENSE file.

## Acknowledgements

### Sample Turbine Data

Plumley, C. (2022). Kelmarsh wind farm data (0.0.3) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5841834