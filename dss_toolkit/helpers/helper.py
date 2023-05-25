# -*- coding: utf-8 -*-
"""
@author: Juan Miguel Recto
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

pd.options.display.float_format = lambda x: f"{x:,.2f}"
import pickle
import pyspark.sql.functions as F


# General functions
def seconds_to_days(x):
    return x / 86400


def days_to_months(days):
    return (days / 365.25) / 12


def days_to_years(days):
    return days / 365.25


def ceiling_div(n, d):
    return -(n // -d)


def n_digits(n):
    "Get the number of digits of an integer"
    return int(np.log10(n - 0.1)) + 1


# Pandas functions


# Plotting functions
def format_ax_ticks(ax, axis="y"):
    "Comma separate ax tick labels."
    if axis == "y":
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
    elif axis == "x":
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )


def rotate_ticklabels(ax, degrees=45):
    "Rotate xticklabels to 45 degrees."
    ax.set_xticklabels(ax.get_xticks(), rotation=degrees, ha="right")


# Analysis helper functions
def cdf(data):
    "Calculate the empirical CDF of a dataset."
    data = pd.Series(data)
    dist = (
        data.value_counts(sort=False)
        .sort_index()
        .cumsum()
        .pipe(lambda x: x / x.iloc[-1])
    )
    return dist


def logbins(data, constant=1, bins=50):
    "Calculate evenly spaced bins in log space."
    _, bin_edges = np.histogram(data + constant, bins=bins)
    logbins = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), bins)
    return logbins


# Spark functions
def filter_by_max(df, filter_col):
    "Filter a Spark DataFrame by the maximum value of a column."
    max_value = df.agg(F.max(filter_col).alias(filter_col))
    df_filtered = df.join(max_value, on=filter_col, how="inner")
    return df_filtered



