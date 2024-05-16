"""
Sample data to use in combination with zephyros.
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
from io import StringIO

import pandas as pd

sample_turbine_data = pd.read_csv(StringIO(
"""
 wind_speed  delta_v  temperature  delta_t  rotor_radius  delta_r  nominal_power  power_coefficient  delta_c  power_expected
     10.587    1.634        7.987    0.039            50      0.5           3400           0.402874      0.0     3400.000000
      7.970    0.959        6.603    0.018            50      0.5           3400           0.513508      0.0        0.000000
      6.383    0.699        0.825    0.034            50      0.5           3400           0.499994      0.0     1057.611726
      7.837    0.857       -1.514    0.172            50      0.5           3400           0.513508      0.0      632.466449
      9.200    1.788        5.418    0.038            50      0.5           3400           0.480939      0.0     2292.144158
      3.939    0.989        7.751    0.046            50      0.5           3400           0.301333      0.0        0.000000
     12.586    1.958        6.866    0.054            50      0.5           3400           0.272416      0.0     3400.000000
     13.660    1.780        7.111    0.029            50      0.5           3400           0.220802      0.0     3109.920171
      8.690    1.318        3.810    0.088            50      0.5           3400           0.494427      0.0     3087.168647
      3.117    0.730        8.016    0.083            50      0.5           3400           0.193215      0.0      687.647267
      7.017    0.704        5.299    0.076            50      0.5           3400           0.513641      0.0        2.845133
     10.490    1.668       21.664    0.078            50      0.5           3400           0.436735      0.0     3400.000000
      7.065    0.719        5.708    0.114            50      0.5           3400           0.513641      0.0     1017.609023
      8.263    1.300       11.425    0.043            50      0.5           3400           0.503260      0.0      906.096673
      9.448    1.185       17.851    0.077            50      0.5           3400           0.480939      0.0     1650.751051
      4.439    0.530       18.283    0.100            50      0.5           3400           0.354126      0.0      255.634778
      7.177    1.310       20.429    0.311            50      0.5           3400           0.513641      0.0        0.000000
      6.193    1.191       13.504    0.087            50      0.5           3400           0.499994      0.0        0.000000
      7.188    0.988       15.204    0.111            50      0.5           3400           0.513641      0.0     1665.824835
      2.483    0.527       18.023    0.053            50      0.5           3400           0.000000      0.0      843.268448
      5.817    1.015       22.211    0.133            50      0.5           3400           0.471444      0.0      607.334282
      1.458    0.310       20.869    0.122            50      0.5           3400           0.000000      0.0     1855.569597
"""
), sep=r"\s+")
