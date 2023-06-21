# -*- coding: utf-8 -*-
"""
@author: Juan Miguel Recto
"""
import numpy as np
import pandas as pd
from IPython.display import display


def show(df, n_decimals=4, thousands_sep=True):
    "Fully display a Pandas DataFrame."
    if thousands_sep:
        thousands_sep = ","
    else:
        thousands_sep = ""

    if n_decimals is not None:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
            "display.float_format",
            f"{{:{thousands_sep}.{n_decimals}f}}".format,
        ):
            display(df)

    else:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
        ):
            display(df)


def show_decimals(df, n_decimals=4):
    with pd.option_context("display.float_format", f"{{:0.{n_decimals}f}}".format):
        display(df)


def set_index(df, index):
    "Rename the index of a Pandas DataFrame."
    df.index = index
    return df


def set_columns(df, columns):
    "Rename the columns of a Pandas DataFrame."
    df.columns = columns
    return df


def deduplicate_names(names, sep="_"):
    "Applend a suffix to duplicate names."
    names = pd.Series(names).astype("str")
    names_deduped = names.values
    for name in names[names.duplicated()].unique():
        dupe_mask = names == name
        n_duplicates = len(names_deduped[dupe_mask])
        n_digits = int(np.log10(n_duplicates - 1)) + 1
        names_deduped[dupe_mask] = [f"{name}{sep}{i:0{n_digits}}" for i in range(n_duplicates)]
    return names_deduped
