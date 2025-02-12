"""
Sample data to use with zephyros.

Citation of the data:
Plumley, C. (2022). Kelmarsh wind farm data (0.0.3) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5841834

Copyright Notice:
The data is licensed under  Creative Commons Attribution 4.0 International.
For more details see (https://creativecommons.org/licenses/by/4.0/legalcode)
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

import matplotlib.pyplot as plt


def plot_prediction(x: list|pd.Index|np.ndarray,
                    pred:tuple, measured:list|pd.Series|np.ndarray,
                    title:str|None=None, xlabel: str|None=None, ylabel: str|None=None) -> tuple:
    """
    Plot a prediction and its uncertainty as well as the measured values.

    Args:
        x (list or pandas.Index or np.ndarray): The x values for the plot.
        pred (tuple): The predicted values and the upper and lower bound of
            their confidence interval. Ideally has the same type as *x*.
        measured (list or pandas.Series or np.ndarray): The measured values.
        title (str or None): The title for the figure. Defaults to None.
        xlabel (str or None): The xlabel for the figure. Defaults to None.
        ylabel (str or None): The ylabel for the figure. Defaults to None.

    Returns:
        tuple: The created figure and axis object.

    """
    y = pred[0]
    fig, ax = plt.subplots()

    ax.scatter(x, measured, color="#0d68b0", s=6, label="expected power",
               zorder=4)
    ax.plot(x, y, color="#d51130", lw=1.2, label="predicted power", zorder=3)
    if len(pred) > 1:
        ly, uy = pred[1:]
        ax.fill_between(x, ly, uy, color="#de9ba7", alpha=1.0, lw=0,
                        label="uncertainty", zorder=1)
        ax.plot(x, ly, color="#872746", lw=0.5, zorder=2, alpha=0.6)
        ax.plot(x, uy, color="#872746", lw=0.5, zorder=2, alpha=0.6)

    legend = ax.legend()
    legend.legend_handles[0].set_sizes([40])
    legend.legend_handles[1].set_lw(2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
