"""Functions for data cleaning.

:author: Andreas Kanz
"""
from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from dss_toolkit.analysis.eda_viz import corr_mat
from dss_toolkit.analysis.utils import _diff_report
from dss_toolkit.analysis.utils import _drop_duplicates


__all__ = [
    "clean_column_names",
    "convert_datatypes",
    "data_cleaning",
    "drop_missing",
    "mv_col_handling",
]


def data_cleaning(
    data: pd.DataFrame,
    drop_threshold_cols: float = 0.9,
    drop_threshold_rows: float = 0.9,
    drop_duplicates: bool = True,
    convert_dtypes: bool = True,
    col_exclude: list[str] | None = None,
    category: bool = True,
    cat_threshold: float = 0.03,
    cat_exclude: list[str | int] | None = None,
    clean_col_names: bool = True,
    show: Literal["all", "changes"] | None = "changes",
) -> pd.DataFrame:
    """Perform initial data cleaning tasks on a dataset.

    For example dropping single valued and empty rows, empty columns as well as \
    optimizing the datatypes.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    drop_threshold_cols : float, optional
        Drop columns with NA-ratio equal to or above the specified threshold, by \
        default 0.9
    drop_threshold_rows : float, optional
        Drop rows with NA-ratio equal to or above the specified threshold, by \
        default 0.9
    drop_duplicates : bool, optional
        Drop duplicate rows, keeping the first occurence. This step comes after the \
        dropping of missing values, by default True
    convert_dtypes : bool, optional
        Convert dtypes using pd.convert_dtypes(), by default True
    col_exclude : Optional[list[str]], optional
        Specify a list of columns to exclude from dropping, by default None
    category : bool, optional
        Enable changing dtypes of "object" columns to "category". Set threshold using \
        cat_threshold. Requires convert_dtypes=True, by default True
    cat_threshold : float, optional
        Ratio of unique values below which categories are inferred and column dtype is \
        changed to categorical, by default 0.03
    cat_exclude : Optional[list[str]], optional
        List of columns to exclude from categorical conversion, by default None
    clean_col_names: bool, optional
        Cleans the column names and provides hints on duplicate and long names, by \
        default True
    show : Optional[Literal["all", "changes"]], optional
        {"all", "changes", None}, by default "changes"
        Specify verbosity of the output:

            * "all": Print information about the data before and after cleaning as \
            well as information about  changes and memory usage (deep). Please be \
            aware, that this can slow down the function by quite a bit.
            * "changes": Print out differences in the data before and after cleaning.
            * None: No information about the data and the data cleaning is printed.

    Returns
    -------
    pd.DataFrame
        Cleaned Pandas DataFrame

    See Also
    --------
    convert_datatypes: Convert columns to best possible dtypes.
    drop_missing : Flexibly drop columns and rows.
    _memory_usage: Gives the total memory usage in megabytes.
    _missing_vals: Metrics about missing values in the dataset.

    Notes
    -----
    The category dtype is not grouped in the summary, unless it contains exactly the \
    same categories.
    """
    if col_exclude is None:
        col_exclude = []

    # Validate Inputs
    _validate_input_range(drop_threshold_cols, "drop_threshold_cols", 0, 1)
    _validate_input_range(drop_threshold_rows, "drop_threshold_rows", 0, 1)
    _validate_input_bool(drop_duplicates, "drop_duplicates")
    _validate_input_bool(convert_dtypes, "convert_datatypes")
    _validate_input_bool(category, "category")
    _validate_input_range(cat_threshold, "cat_threshold", 0, 1)

    data = pd.DataFrame(data).copy()
    data_cleaned = drop_missing(
        data,
        drop_threshold_cols,
        drop_threshold_rows,
        col_exclude=col_exclude,
    )

    if clean_col_names:
        data_cleaned = clean_column_names(data_cleaned)

    single_val_cols = data_cleaned.columns[data_cleaned.nunique(dropna=False) == 1].tolist()
    single_val_cols = [col for col in single_val_cols if col not in col_exclude]
    data_cleaned = data_cleaned.drop(columns=single_val_cols)

    dupl_rows = None

    if drop_duplicates:
        data_cleaned, dupl_rows = _drop_duplicates(data_cleaned)
    if convert_dtypes:
        data_cleaned = convert_datatypes(
            data_cleaned,
            category=category,
            cat_threshold=cat_threshold,
            cat_exclude=cat_exclude,
        )

    _diff_report(
        data,
        data_cleaned,
        dupl_rows=dupl_rows,
        single_val_cols=single_val_cols,
        show=show,
    )

    return data_cleaned


def pool_duplicate_subsets(
    data: pd.DataFrame,
    col_dupl_thresh: float = 0.2,
    subset_thresh: float = 0.2,
    min_col_pool: int = 3,
    exclude: list[str] | None = None,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Check for duplicates in subsets of columns and pools them.

    This can reduce the number of columns in the data without loosing much \
    information. Suitable columns are combined to subsets and tested for duplicates. \
    In case sufficient duplicates can be found, the respective columns are aggregated \
    into a "pooled_var" column. Identical numbers in the "pooled_var" column indicate \
    identical information in the respective rows.

    Note:  It is advised to exclude features that provide sufficient informational \
    content by themselves as well as the target column by using the "exclude" \
    setting.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    col_dupl_thresh : float, optional
        Columns with a ratio of duplicates higher than "col_dupl_thresh" are \
        considered in the further analysis. Columns with a lower ratio are not \
        considered for pooling, by default 0.2
    subset_thresh : float, optional
        The first subset with a duplicate threshold higher than "subset_thresh" is \
        chosen and aggregated. If no subset reaches the threshold, the algorithm \
        continues with continuously smaller subsets until "min_col_pool" is reached, \
        by default 0.2
    min_col_pool : int, optional
        Minimum number of columns to pool. The algorithm attempts to combine as many \
        columns as possible to suitable subsets and stops when "min_col_pool" is \
        reached, by default 3
    exclude : Optional[list[str]], optional
        List of column names to be excluded from the analysis. These columns are \
        passed through without modification, by default None
    return_details : bool, optional
        Provdies flexibility to return intermediary results, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with low cardinality columns pooled

    optional:
    subset_cols: List of columns used as subset
    """
    # Input validation
    _validate_input_range(col_dupl_thresh, "col_dupl_thresh", 0, 1)
    _validate_input_range(subset_thresh, "subset_thresh", 0, 1)
    _validate_input_range(min_col_pool, "min_col_pool", 0, data.shape[1])

    excluded_cols = []
    if exclude is not None:
        excluded_cols = data[exclude]
        data = data.drop(columns=exclude)

    subset_cols = []
    for i in range(data.shape[1] + 1 - min_col_pool):
        # Consider only columns with lots of duplicates
        check = [col for col in data.columns if data.duplicated(subset=col).mean() > col_dupl_thresh]

        # Identify all possible combinations for the current interation
        if check:
            combinations = itertools.combinations(check, len(check) - i)
        else:
            continue

        # Check subsets for all possible combinations
        ratios = [*(data.duplicated(subset=list(comb)).mean() for comb in combinations)]

        max_idx = np.argmax(ratios)

        if max(ratios) > subset_thresh:
            # Get the best possible iterator and process the data
            best_subset = itertools.islice(
                itertools.combinations(check, len(check) - i),
                max_idx,
                max_idx + 1,
            )

            best_subset = data[list(list(best_subset)[0])]
            subset_cols = best_subset.columns.tolist()

            unique_subset = best_subset.drop_duplicates().reset_index().rename(columns={"index": "pooled_vars"})
            data = data.merge(unique_subset, how="left", on=subset_cols).drop(
                columns=subset_cols,
            )
            data.index = pd.RangeIndex(len(data))
            break

    data = pd.concat([data, pd.DataFrame(excluded_cols)], axis=1)

    return (data, subset_cols) if return_details else data
