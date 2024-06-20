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
import pandas as pd


def madtm(actual, predicted, lower, upper):
    """

    Args:
        actual (pd.Series): The measured values.
        predicted (pd.Series): The predicted values.
        lower (pd.Series): The lower bound of the confidence interval for the
            predicted values.
        upper (pd.Series): The upper bound of the confidence interval for the
            predicted values.

    Returns:
        pd.Series: The mean absolute distance to margin. A quantification of the
            quality of the confidence interval.

    """
    distance_to_lower = (actual - lower).abs()
    distance_to_upper = (actual - upper).abs()
    min_distance = pd.Series(data=0.0, index=actual.index)
    use_lower = distance_to_lower < distance_to_upper
    min_distance.loc[use_lower] = distance_to_lower.loc[use_lower]
    min_distance.loc[~use_lower] = distance_to_upper.loc[~use_lower]
    return min_distance.abs().mean()


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