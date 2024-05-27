"""
Module for physical prediction of the estimated power output of a wind turbine
based on environmental parameters such as wind speed and ambient temperature.
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


def predict(x):
    """
    Perform physical calculations to predict the power output of a wind turbine
    given ambient temperature $T$ in Degree Celsius, rotor radius $R$ in m,
    capacity factor $c_p$, wind speed $v$ in m/s, nominal power $P_max$ in kW
    and their respective uncertainties.

    Args:
        x(dict or pd.DataFrame): Parameters required for the calculation of
            the power output. Can either be a dictionary containing float or
            time series data as numpy arrays, or a pandas DataFrame.
            Required keys are:
                - *temperature*: The temperature in Degree Celsius
                - *delta_t*: The uncertainty of the temperature in Degree Celsius.
                - *rotor_radius*: The rotor radius in m.
                - *delta_r*: The uncertainty of the rotor radius in m.
                - *capacity_factor*: The capacity factor.
                - *delta_c*: The uncertainty of the capacity factor..
                - *wind_speed*: The wind speed at hub height in m/s.
                - *delta_v*: The uncertainty of the wind speed in m/s.
                - *nominal_power*: The nominal power of the turbine in kW.

    Returns:
        tuple: The predicted power in kW and its estimated uncertainty.

    """
    _catch_missing_keys(x)
    rho, delta_rho = _calculate_rho(x["temperature"], x["delta_t"])
    power = _calculate_power(x["rotor_radius"], x["capacity_factor"],
                             rho, x["wind_speed"], x["nominal_power"])
    delta_power = _calculate_delta_power(
        x["rotor_radius"], x["capacity_factor"], rho, x["wind_speed"],
        x["delta_r"], x["delta_c"], delta_rho, x["delta_v"]
    )
    return power, delta_power


def _catch_missing_keys(x):
    """
    Check if all necessary keys for the calculations are present and, if not,
    raise a key error specifying which keys are missing.

    Args:
        x(dict or pd.DataFrame): Parameters required for the calculation of
            the power output. Required keys are:

                - *temperature*: The temperature in Degree Celsius
                - *delta_t*: The uncertainty of the temperature in Degree Celsius.
                - *rotor_radius*: The rotor radius in m.
                - *delta_r*: The uncertainty of the rotor radius in m.
                - *capacity_factor*: The capacity factor.
                - *delta_c*: The uncertainty of the capacity factor.
                - *wind_speed*: The wind speed at hub height in m/s.
                - *delta_v*: The uncertainty of the wind speed in m/s.
                - *nominal_power*: The nominal power of the turbine in kW.

    Returns:
        None.

    Raises:
        KeyError: If any of the required keys are missing.

    """
    required = {
        "temperature": "'temperature' - The temperature in Degree Celsius",
        "delta_t": "'delta_t' - The uncertainty of the temperature in Degree Celsius.",
        "rotor_radius": "'rotor_radius' - The rotor radius in m.",
        "delta_r": "'delta_r' - The uncertainty of the rotor radius in m.",
        "capacity_factor": "'capacity_factor' - The capacity factor.",
        "delta_c": "'delta_c' - The uncertainty of the capacity factor.",
        "wind_speed": "'wind_speed' - The wind speed at hub height in m/s.",
        "delta_v": "'delta_v' - The uncertainty of the wind speed in m/s.",
        "nominal_power": "'nominal_power' - The nominal power of the turbine in kW"
    }
    missing = []

    for key in required:
        if key not in x:
            missing.append(required[key])

    if missing:
        err_msg = "Missing required keys for calculation:\n"
        err_msg += "\n".join(missing)
        raise KeyError(err_msg)


def _calculate_rho(temperature, delta_t):
    """
    Approximate the air density rho by linearly interpolating between the values
    given in reference [1]. Air pressure is assumed to be 1 bar and the air is
    considered dry.

    [1] VDI-Wärmeatlas, 8th ed. in VDI-Buch. Heidelberg: Springer Berlin, 1997.
    p. 183. Accessed: May 06, 2024. [Online]
    Available: https://doi.org/10.1007/978-3-662-10745-4.

    Args:
        temperature (float or np.ndarray or pd.Series): Temperature in Degree Celsius.
        delta_t (float or pd.Series): The uncertainty of the temperature
            in Degree Celsius.

    Returns:
        tuple: The calculated air density in kg/m^3 and the uncertainty in kg/m^3.

    """
    if isinstance(temperature, float):
        return _calculate_single_rho(temperature, delta_t)
    df = pd.DataFrame({"x": temperature, "ux": delta_t})
    result = df.apply(lambda x: _calculate_single_rho(x["x"], x["ux"]), axis=1)
    rho, delta_rho = zip(*result)
    return pd.Series(rho), pd.Series(delta_rho)


def _calculate_single_rho(temperature, delta_t):
    """
    Approximate the air density rho by linearly interpolating between the values
    given in reference [1]. Air pressure is assumed to be 1 bar and the air is
    considered dry.

    [1] VDI-Wärmeatlas, 8th ed. in VDI-Buch. Heidelberg: Springer Berlin, 1997.
    p. 183. Accessed: May 06, 2024. [Online]
    Available: https://doi.org/10.1007/978-3-662-10745-4.

    Args:
        temperature (float): Temperature in Degree Celsius.
        delta_t (float): The uncertainty of the temperature in Degree Celsius.

    Returns:
        tuple: The calculated air density in kg/m^3 and the uncertainty in kg/m^3.

    """
    # piecewise linear approximation of rho
    # with known values for -50, -25, 0, 25, 50 °C respectively
    if -50.0 <= temperature < -25.0:
        rho = 1.563 + (1.404 - 1.563) * (temperature + 50) / 25
        delta_rho = delta_t * np.abs((1.404 - 1.563) / 25)
    elif -25.0 <= temperature < 0.0:
        rho = 1.404 + (1.275 - 1.404) * (temperature + 25) / 25
        delta_rho = delta_t * np.abs((1.275 - 1.404) / 25)
    elif 0.0 <= temperature < 25.0:
        rho = 1.275 + (1.168 - 1.275) * (temperature + 0) / 25
        delta_rho = delta_t * np.abs((1.168 - 1.275) / 25)
    elif 25.0 <= temperature < 50.0:
        rho = 1.168 + (1.078 - 1.168) * (temperature - 25) / 25
        delta_rho = delta_t * np.abs((1.078 - 1.168) / 25)
    else:
        raise NotImplementedError(
            f"Temperature {temperature} °C is outside the calculation range!\n"
            "Temperature must be in the half-open interval "
            "[-50.0 °C, 50 °C)."
        )
    return rho, delta_rho


def _calculate_power(rotor_radius, capacity_factor, air_density, wind_speed,
                     nominal_power):
    """
    Calculate the power output with the formula given in reference [1].

    [1] University of Leipzig, ‘Physics of Wind Turbines | Energy Fundamentals’.
     Accessed: May 06, 2024. [Online].
     Available: https://home.uni-leipzig.de/energy/energy-fundamentals/15.htm

    Args:
        rotor_radius(float): The radius $R$ of the turbine's blades in m.
        capacity_factor(float or np.ndarray or pd.Series): The capacity factor $c_p$.
        air_density(float or np.ndarray or pd.Series): The air density $\rho$ in kg/m^3.
        wind_speed(float or np.ndarray or pd.Series): The wind speed $v$ in m/s.
        nominal_power(float): The nominal power of the wind turbine in kW.

    Returns:
        float or np.ndarray or pd.Series: The predicted power output in kW based
            on the calculations.

    """
    wind_area = np.pi * rotor_radius ** 2
    power = 1 / 2 * capacity_factor * air_density * wind_area * wind_speed ** 3
    # convert W to kW
    power /= 1_000
    if isinstance(power, float):
        power = power if power < nominal_power else nominal_power
    else:
        power[power > nominal_power] = nominal_power
    return power


def _calculate_delta_power(rotor_radius, capacity_factor, air_density,
                           wind_speed, delta_r, delta_c, delta_rho, delta_v):
    """
    Calculate the uncertainty of the power output prediction using the rules
    of uncertainty propagation.

    Args:
        rotor_radius(float): The radius $R$ of the turbine's blades in m.
        capacity_factor(float or np.ndarray or pd.Series): The capacity factor $c_p$.
        air_density(float or np.ndarray or pd.Series): The air density $\rho$ in kg/m^3.
        wind_speed(float or np.ndarray or pd.Series): The wind speed $v$ in m/s.
        delta_r(float): The uncertainty of the rotor radius $R$ in m.
        delta_c(float or np.ndarray or pd.Series): The uncertainty of the power
            coefficient $c_p$.
        delta_rho(float or np.ndarray or pd.Series): The uncertainty of the air
            density $rho$ in kg/m^3.
        delta_v(float or np.ndarray or pd.Series): The uncertainty of the wind
            speed $v$ in m/s.

    Returns:
        float or np.ndarray or pd.Series: The uncertainty of the calculated power
            output.

    """
    wind_area = np.pi * rotor_radius ** 2
    delta_power = (
        delta_r * np.abs(np.pi * rotor_radius * capacity_factor * air_density * wind_speed ** 3)
    +   delta_c * np.abs(1 / 2 * air_density * wind_area * wind_speed ** 3)
    +   delta_rho * np.abs(1 / 2 * capacity_factor * wind_area * wind_speed ** 3)
    +   delta_v * np.abs(3 / 2 * capacity_factor * air_density * wind_area * wind_speed ** 2)
    )
    # convert W to kW
    delta_power /= 1_000
    return delta_power
