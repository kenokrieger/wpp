"""
Tests for the zephyros.empirical_predictor module.
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
from itertools import chain, combinations

from zephyros.empirical_predictor import learn_and_predict

TEST_DATA = "./tests/test_data/test_data.csv"
TREE_TEST_CASES = list(chain.from_iterable(
    combinations(["wind_speed", "temperature", "delta_v", "delta_t"], r)
    for r in range(1, 5))
)
TREE_TEST_ACCURACIES = [1, 2, 3, 4, 8, 16]


@pytest.mark.parametrize('test_case', TREE_TEST_CASES)
@pytest.mark.parametrize('acc', TREE_TEST_ACCURACIES)
def test_learn_and_predict(test_case, acc):
    """
    Test the learn_and_predict method from the zephyros.empirical_predictor
    module.

    Args:
        test_case(iterable): The features to use for the prediction.
        acc(int): The accuracy of the prediction.

    """
    x = pd.read_csv(TEST_DATA)
    learn_data = x.loc[:34_000]
    predict_data = x.loc[34_000:]

    if len(test_case) == 4 and acc == 16:
        with pytest.raises(ValueError):
            y = learn_and_predict(learn_data, predict_data, test_case,
                                  ["power_measured"], acc)
    else:
        y = learn_and_predict(learn_data, predict_data, test_case,
                              ["power_measured"], acc)
        name = f"./tests/test_data/{'_'.join((t for t in test_case))}_{acc}.csv"
        expected = pd.read_csv(name)
        assert (np.allclose(y["predicted"], expected["predicted"], rtol=0.02)
                and
                np.allclose(y["uncertainty"], expected["uncertainty"], rtol=0.02))
