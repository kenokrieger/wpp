"""
Tests for the zephyros.physical_predictor module.
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
import pandas as pd
import pytest

from zephyros.physical_predictor import predict, _catch_missing_keys, _calculate_rho

REQUIRED_KEYS = ["temperature", "delta_t", "rotor_radius", "delta_r",
                 "power_coefficient", "delta_c", "wind_speed", "delta_v",
                 "nominal_power"]
KEY_TEST_CASES = [{k: None for k in REQUIRED_KEYS[:m]}
                  for m in range(0, len(REQUIRED_KEYS))]
RHO_TEST_CASES = [
    [(-50.0, 0.0), (1.563, 0.0)],
    [(-25.0, 0.0), (1.404, 0.0)],
    [(0.0, 0.0), (1.275, 0.0)],
    [(25.0, 0.0), (1.168, 0.0)],
    [(49.99, 0.0), (1.078, 0.0)],
    [(np.array([-50.0, -25.0, 0.0, 25.0, 49.99]), 0.0),
     (np.array([1.563, 1.404, 1.275, 1.168, 1.078]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]))],
    [(np.array([-50.0, -25.0, 0.0, 25.0, 49.99]), np.array([0.5, 1.2, 0.1, 25.0, 1.0])),
     (np.array([1.563, 1.404, 1.275, 1.168, 1.078]), np.array([0.00318, 0.006192, 0.000428, 0.09, 0.0036]))],
    [(pd.Series([-10.3, 5.0, 7.3, 28.6, 3.7]), 0.0),
     (pd.Series([1.328, 1.254, 1.244, 1.155, 1.259]), pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))],
    [(pd.Series([-10.3, 5.0, 7.3, 28.6, 3.7]), pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])),
     (pd.Series([1.328, 1.254, 1.244, 1.155, 1.259]), pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))]
]
RHO_RAISE_CASES = [
    -125.0, np.array([-1.0, 52.3, 45.1]), pd.Series([23652.1, 0.1, 1.1])
]
PREDICT_CASES = ["./tests/test_data/test_data.csv"]


@pytest.mark.parametrize('test_case', KEY_TEST_CASES)
def test_catch_missing_keys_raises(test_case):
    """
    Tests that a KeyError is raised if input is missing keys.

    Args:
        test_case (dict): A dictionary missing required keys.

    """
    with pytest.raises(KeyError) as e_info:
        _catch_missing_keys(test_case)


def test_catch_missing_keys_passes():
    """
    Tests that no error is raised when all keys are present in the input.
    """
    assert _catch_missing_keys(REQUIRED_KEYS) is None


@pytest.mark.parametrize('test_case', RHO_TEST_CASES)
def test_calculate_rho(test_case):
    """
    Tests that the correct values for the air density are returned.

    Args:
        test_case (float or np.ndarray or pd.Series): Input and expected value(s)
            for the temperature and density respectively.

    """
    test_in = test_case[0]
    test_out = test_case[1]
    rho, delta_rho = _calculate_rho(test_in[0], test_in[1])
    assert (
            np.allclose(rho, test_out[0], rtol=0.01) and
            np.allclose(delta_rho, test_out[1], rtol=0.01)
    )


@pytest.mark.parametrize('test_case', RHO_RAISE_CASES)
def test_calculate_rho_raises(test_case):
    """
    Tests that an error is raised if the temperature is not in the
    expected range.

    Args:
        test_case (float or np.ndarray or pd.Series): Value(s) for the temperature.
    """
    with pytest.raises(NotImplementedError):
        _calculate_rho(test_case, 0.0)


@pytest.mark.parametrize('test_case', PREDICT_CASES)
def test_predict(test_case):
    """
    Test the predict function of the physical_predictor module.

    Args:
        test_case(str): The path to the file containing test data.

    """
    x = pd.read_csv(test_case)
    y, uy = predict(x)
    assert (np.allclose(y, x["power_expected"], rtol=0.02) and
            np.allclose(uy, x["delta_power_expected"], rtol=0.02))
