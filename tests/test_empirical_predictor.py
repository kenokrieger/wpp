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

from zephyros.empirical_predictor import grow_tree

TEST_DATA = "./tests/test_data/test_data.csv"
TREE_TEST_CASES = [chain.from_iterable(
    combinations(["wind_speed", "temperature", "delta_v", "delta_t"], r)
    for r in range(5))
][1:]  # remove empty list from test cases
TREE_TEST_ACCURACIES = list(range(1, 11))


@pytest.mark.parametrize('test_case', TREE_TEST_CASES)
@pytest.mark.parametrize('acc', TREE_TEST_ACCURACIES)
def test_grow_tree(test_case, acc):
    """
    Test the grow_tree function from the zephyros.empirical_predictor module.

    Args:
        test_case(iterable): The features to use for the tree creation.
        acc(int): The accuracy of the tree.

    """
    x = pd.read_csv(TEST_DATA)
    y, uy = grow_tree(x, test_case,"power_measured", acc)
    # expected four paths, four values
    print(y)
    print(uy)
    assert len(y) == 8
