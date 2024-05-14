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
from zephyros.sample_data import sample_turbine_data

x = sample_turbine_data
x["predicted_power"], x["predicted_power_uncertainty"] = predict(x)

fig, ax = plt.subplots()
plt.plot(x.index, x["power_expected"], label="expected power")
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
## Development

### Tests

**IMPORTANT NOTICE: Test data required for some of the tests
is not openly available!**

Tests can be run by executing `python3 -m pytest` in the highest
directory.

### License

This project is licensed under GNU GENERAL PUBLIC LICENSE.
For more details see the LICENSE file.

### Acknowledgements
