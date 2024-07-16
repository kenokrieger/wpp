"""
zephyros, named after one of the Greek gods of wind,
is a package designed for predicting power output from wind turbines
(or wind farms) using different methods ranging from physical power
calculations to machine learning.
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
from zephyros import physical_predictor
from zephyros import empirical_predictor
from zephyros import svm_predictor
from zephyros import rvm_predictor
from zephyros import boost_predictor
from zephyros import ann_predictor
from zephyros import examples
