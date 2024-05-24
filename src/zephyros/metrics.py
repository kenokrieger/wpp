"""
Module containing evaluation metrics to assess model quality and forecast
uncertainty quality.
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


def uncertainty_quality(actual, predicted, uncertainty):
    """
    Assess the quality of the uncertainty by comparing it to the actual
    deviation of the prognosis from the measured values.

    Args:
        actual (float or np.ndarray or pd.Series): The measured values.
        predicted (float or np.ndarray or pd.Series): The predicted values.
        uncertainty (float or np.ndarray or pd.Series): The uncertainty of the
            predicted values.

    Returns:
        float: The total sum of the deviation from the uncertainty to the
            measured deviation.

    """
    uncertainty_measured = np.abs(actual - predicted)
    difference = uncertainty - uncertainty_measured
    # small positive is good
    # large negative is bad
    return np.sum(difference)


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))