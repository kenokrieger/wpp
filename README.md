# <img src="https://raw.githubusercontent.com/kenokrieger/zephyros/master/LOGO.png" alt="zephyros" height=60>

zephyros, named after one of the Greek gods of wind,
is a package designed for predicting power output from wind turbines
(or wind farms) using different methods ranging from physical power 
calculations to machine learning.

## Install

Download the source code of the newest release from
[GitHub](https://github.com/kenokrieger/zephyros). Then run
```bash
python3 -m pip install .
``` 
in the  highest directory.

## Usage

### Examples

#### Example 1: Use the physical prediction method

```python
import matplotlib.pyplot as plt

from zephyros.physical_predictor import predict
from zephyros.examples import get_sample_data, plot_prediction

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
import matplotlib.pyplot as plt

from zephyros.empirical_predictor import (learn_and_predict, PREDICT_KEY,
                                          UNCERTAINTY_KEY)
from zephyros.examples import get_sample_data, plot_prediction

x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.99 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = "power_measured"
y = learn_and_predict(learn_data, predict_data,
                      features, target, accuracy=12)
plot_values = (y, y[PREDICT_KEY] - y[UNCERTAINTY_KEY],
               y[PREDICT_KEY] + y[UNCERTAINTY_KEY])
fig, ax = plot_prediction(y.index, plot_values, 
                          predict_data["power_measured"],
                          title="Power Prediction based on Historically Measured Values",
                          xlabel="Time index", ylabel="Power in kW")
plt.show()
```

#### Example 3: Use Support Vector Regression
```python
import matplotlib.pyplot as plt

from zephyros.svm_predictor import learn_and_predict
from zephyros.examples import get_sample_data, plot_prediction

x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
x_in = learn_data[features].to_numpy()
y_in = learn_data[target].to_numpy().ravel()
x_pred = predict_data[features].to_numpy()
model = svm_predictor.learn(x_in, y_in)
y = svm_predictor.predict(model, x_pred)
estimators = model.estimators_
estimates = np.array([
    e.predict(x_pred) for e in estimators
])
std_y = estimates.std(axis=0)
fig, ax = plot_prediction(predict_data.index, (y, y - std_y / 2, y + std_y / 2),
                          predict_data["power_measured"],
                          title="Power Prediction using SVR",
                          xlabel="Time index", ylabel="Power in kW")
plt.show()
```

#### Example 4: Use Relevance Vector Regression
```python
import matplotlib.pyplot as plt

from zephyros.rvm_predictor import learn_and_predict
from zephyros.examples import get_sample_data, plot_prediction

x = get_sample_data().iloc[:2_000]
nrows = x.shape[0]
learn_predict_split = int(0.95 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y, std_y = learn_and_predict(learn_data, predict_data, features, target)

fig, ax = plot_prediction(predict_data.index,
                          (y, y - std_y / 2, y + std_y / 2),
                          predict_data["power_measured"],
                          title="Power Prediction using RVM",
                          xlabel="Time index", ylabel="Power in kW")

plt.show()
```

#### Example 5: Use Extreme Gradient Boosting (xgboost package)
```python
import matplotlib.pyplot as plt

from zephyros.boost_predictor import learn_and_predict
from zephyros.examples import get_sample_data, plot_prediction

x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
# predict with 8 cross validations
y, ly, uy = learn_and_predict(learn_data, predict_data, features, target,
                              xvalidate=8)
# average the results of each cross validation
y = y.mean(axis=0)
ly = ly.mean(axis=0)
uy = uy.mean(axis=0)
plot_prediction(y, ly, uy, predict_data)
plt.show()
```

#### Example 6: Use Artificial Neural Networks
```python
import matplotlib.pyplot as plt

from zephyros.ann_predictor import learn_and_predict
from zephyros.examples import get_sample_data, plot_prediction

x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y = learn_and_predict(learn_data, predict_data, features, target, xvalidate=4)
y = y.mean(axis=0)
fig, ax = plot_prediction(predict_data.index, (y, y, y), predict_data["power_measured"],
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
(e.g. boost_predictor.py) are not tested.

## License

This project is licensed under GNU GENERAL PUBLIC LICENSE.
For more details see the LICENSE file.

## Acknowledgements

### Sample Turbine Data

Plumley, C. (2022). Kelmarsh wind farm data (0.0.3) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5841834
