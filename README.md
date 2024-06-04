# zephyros

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
from zephyros.sample_data import get_sample_data

x = get_sample_data().sort_index()
# use 200 sample values
x = x.iloc[10_000:10_200]
# calculate the expected power output of a wind turbine
# based on wind speed, temperature and capacity factor
x["predicted_power"], x["predicted_power_uncertainty"] = predict(x)

# visualise the results
fig, ax = plt.subplots()
plt.plot(x.index, x["power_measured"], label="measured power")
plt.plot(x.index, x["predicted_power"], label="predicted power")
plt.fill_between(
    x.index, x["predicted_power"] + x["predicted_power_uncertainty"],
             x["predicted_power"] - x["predicted_power_uncertainty"],
    color="orange", alpha=0.3, linewidth=0, label="uncertainty")
plt.legend()
ax.set_title("Power Prediction based on Physical Calculations")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
plt.show()
```

### Example 2: Use the empirical prediction method

```python
import matplotlib.pyplot as plt

from zephyros.empirical_predictor import (learn_and_predict, PREDICT_KEY, 
                                          UNCERTAINTY_KEY)
from zephyros.sample_data import get_sample_data

x = get_sample_data()
nrows = x.shape[0]
# use 99 % of data for "learning"
learn_predict_split = int(0.99 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
# predict the power output of a wind turbine based on
# historical values of wind speed and temperature
# and the respective resulting power generation of
# the turbine
features = ["wind_speed", "temperature"]
target = "power_measured"
y = learn_and_predict(learn_data, predict_data,
                      features, target, accuracy=12)
# visualise the results
fig, ax = plt.subplots()
plt.plot(y.index, predict_data["power_measured"], label="expected power")
plt.plot(y.index, y[PREDICT_KEY], label="predicted power")
plt.fill_between(
    y.index, y[PREDICT_KEY] + y[UNCERTAINTY_KEY],
    y[PREDICT_KEY] - y[UNCERTAINTY_KEY],
    color="orange", alpha=0.3, linewidth=0, label="uncertainty")
plt.legend()
ax.set_title("Power Prediction based on Historically Measured Values")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
plt.show()
```

### Example 3: Use Support Vector Regression
```python
import matplotlib.pyplot as plt

from zephyros.svm_predictor import learn_and_predict
from zephyros.sample_data import get_sample_data

x = get_sample_data()
nrows = x.shape[0]
# use 99.9 % of data for learning
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
# predict the power output of a wind turbine based on
# historical values of wind speed and temperature
# and the respective resulting power generation of
# the turbine
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y = svm_predictor.learn_and_predict(learn_data, predict_data,
                                    features, target)
fig, ax = plt.subplots()
plt.scatter(predict_data.index, predict_data["power_measured"],
            label="expected power")
plt.plot(predict_data.index, y, label="predicted power")
plt.legend()
ax.set_title("Power Prediction using SVR")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
plt.show()
```

### Example 4: Use Relevance Vector Regression
```python
import matplotlib.pyplot as plt

from zephyros.rvm_predictor import learn_and_predict
from zephyros.sample_data import get_sample_data

# Use only a fraction of sample data to avoid memory issues
x = get_sample_data().iloc[:2_000]
nrows = x.shape[0]
# use 99 % of data for learning
learn_predict_split = int(0.95 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
# predict the power output of a wind turbine based on
# historical values of wind speed and temperature
# and the respective resulting power generation of
# the turbine
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y, std_y = learn_and_predict(learn_data, predict_data, features, target)

fig, ax = plt.subplots()
plt.scatter(predict_data.index, predict_data["power_measured"],
            label="expected power")
plt.plot(predict_data.index, y, label="predicted power")
plt.fill_between(predict_data.index, y + std_y / 2, y - std_y / 2,
                 color="orange", alpha=0.3, linewidth=0,
                 label="uncertainty")
plt.legend()
ax.set_title("Power Prediction using RVM")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
plt.show()
```

### Example 5: Use Extreme Gradient Boosting (xgboost package)
```python
import matplotlib.pyplot as plt

from zephyros.boost_predictor import learn_and_predict
from zephyros.sample_data import get_sample_data

x = get_sample_data()
nrows = x.shape[0]
# use 99 % of data for learning
learn_predict_split = int(0.99 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
# predict the power output of a wind turbine based on
# historical values of wind speed and temperature
# and the respective resulting power generation of
# the turbine using extreme gradient boosting
features = ["wind_speed", "temperature"]
target = ["power_measured"]
# predict with 16 cross validations
y, ly, uy = learn_and predict(learn_data, predict_data, features, target,
                              xvalidate=16)
# average the results of each cross validation
y, ly, uy = y.mean(axis=0), ly.mean(axis=0), uy.mean(axis=0)
# visualise the results
fig, ax = plt.subplots()
plt.plot(predict_data.index, predict_data["power_measured"], label="expected power")
plt.plot(predict_data.index, y, label="predicted power")
plt.fill_between(predict_data.index, uy, ly, color="orange", alpha=0.3, linewidth=0,
                 label="uncertainty")
plt.legend()
ax.set_title("Power Prediction using Extreme Gradient Boosting")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
plt.show()
```

### Example 6: Use Artificial Neural Networks
```python
import matplotlib.pyplot as plt

from zephyros.ann_predictor import learn_and_predict
from zephyros.sample_data import get_sample_data

x = get_sample_data()
nrows = x.shape[0]
learn_predict_split = int(0.999 * nrows)
learn_data = x.iloc[:learn_predict_split]
predict_data = x.iloc[learn_predict_split:]
features = ["wind_speed", "temperature"]
target = ["power_measured"]
y = ann_learn_and_predict(learn_data, predict_data, features, target,
                          xvalidate=4)
y = y.mean(axis=0)
# visualise the results
fig, ax = plt.subplots()
plt.plot(predict_data.index, predict_data["power_measured"],
         label="expected power")
plt.plot(predict_data.index, y, label="predicted power")
plt.legend()
ax.set_title("Power Prediction with ANN")
ax.set_xlabel("Time index")
ax.set_ylabel("Power in kW")
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
