"""Utilities and auxiliary functions.

:author: Andreas Kanz

"""
from __future__ import annotations

from typing import Literal


import numpy as np
import pandas as pd


def _diff_report(
    data: pd.DataFrame,
    data_cleaned: pd.DataFrame,
    dupl_rows: list[str | int] | None = None,
    single_val_cols: list[str] | None = None,
    show: Literal["all", "changes"] | None = "changes",
) -> None:
    """Provide information about changes between two datasets.

    This includes dropped rows and columns, memory usage and missing values.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. Input the initial \
        dataset here
    data_cleaned : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. Input the cleaned / \
        updated dataset here
    dupl_rows : Optional[list[str | int]], optional
        List of duplicate row indices, by default None
    single_val_cols : Optional[List[str]], optional
        List of single-valued column indices. I.e. columns where all cells contain \
        the same value. NaNs count as a separate value, by default None
    show : str, optional
        {"all", "changes", None}, by default "changes"
        Specify verbosity of the output:
            * "all": Print information about the data before and after cleaning as \
                well as information about changes and memory usage (deep). Please be \
                aware, that this can slow down the function by quite a bit.
            * "changes": Print out differences in the data before and after cleaning.
            * None: No information about the data and the data cleaning is printed.

    Returns
    -------
    None
        Print statement highlighting the datasets or changes between the two datasets.
    """
    if show not in ["changes", "all"]:
        return

    dupl_rows = [] if dupl_rows is None else dupl_rows.copy()
    single_val_cols = [] if single_val_cols is None else single_val_cols.copy()
    data_mem = _memory_usage(data, deep=False)
    data_cl_mem = _memory_usage(data_cleaned, deep=False)
    data_mv_tot = _missing_vals(data)["mv_total"]
    data_cl_mv_tot = _missing_vals(data_cleaned)["mv_total"]

    if show == "all":
        data_mem = _memory_usage(data, deep=True)
        data_cl_mem = _memory_usage(data_cleaned, deep=True)
        _print_cleaning_details("Before data cleaning:\n", data, data_mv_tot, data_mem)
        _print_cleaning_details(
            "After data cleaning:\n",
            data_cleaned,
            data_cl_mv_tot,
            data_cl_mem,
        )

    print(
        f"Shape of cleaned data: {data_cleaned.shape} - " f"Remaining NAs: {data_cl_mv_tot}\n\n",
    )
    print(f"Dropped rows: {data.shape[0]-data_cleaned.shape[0]}")
    print(
        f"     of which {len(dupl_rows)} duplicates. (Rows (first 150 shown): {dupl_rows[:150]})\n",  # noqa: E501
    )
    print(f"Dropped columns: {data.shape[1]-data_cleaned.shape[1]}")
    print(
        f"     of which {len(single_val_cols)} single valued." f"     Columns: {single_val_cols}",
    )
    print(f"Dropped missing values: {data_mv_tot-data_cl_mv_tot}")
    mem_change = data_mem - data_cl_mem
    mem_perc = round(100 * mem_change / data_mem, 2)
    print(f"Reduced memory by at least: {round(mem_change,3)} MB (-{mem_perc}%)\n")


def _print_cleaning_details(
    header: str,
    data: pd.DataFrame | pd.Series,
    missing_vals: int,
    mem_usage: float,
) -> None:
    print(header)
    print(f"dtypes:\n{data.dtypes.value_counts()}")
    print(f"\nNumber of rows: {str(data.shape[0]).rjust(8)}")
    print(f"Number of cols: {str(data.shape[1]).rjust(8)}")
    print(f"Missing values: {str(missing_vals).rjust(8)}")
    print(f"Memory usage: {str(mem_usage).rjust(7)} MB")
    print("_______________________________________________________\n")


def _drop_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str | int]]:
    """Provide information on and drops duplicate rows.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, List]
        Deduplicated Pandas DataFrame and Index Object of rows dropped
    """
    data = pd.DataFrame(data).copy()
    dupl_rows = data[data.duplicated()].index.tolist()
    data = data.drop(dupl_rows, axis="index").reset_index(drop=True)

    return data, dupl_rows


def _memory_usage(data: pd.DataFrame, deep: bool = True) -> float:
    """Give the total memory usage in megabytes.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    deep : bool, optional
        Runs a deep analysis of the memory usage, by default True

    Returns
    -------
    float
        Memory usage in megabytes
    """
    return round(data.memory_usage(index=True, deep=deep).sum() / (1024**2), 2)
