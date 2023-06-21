from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

from dss_toolkit.base.input_validation import _validate_input_range, _validate_input_bool, _validate_input_num_data


def corr_mat(
    data: pd.DataFrame,
    split: Literal["pos", "neg", "high", "low"] | None = None,
    threshold: float = 0,
    target: pd.DataFrame | pd.Series | np.ndarray | str | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    colored: bool = True,
) -> pd.DataFrame | pd.Series:
    """Return a color-encoded correlation matrix.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    split : Optional[Literal['pos', 'neg', 'high', 'low']], optional
        Type of split to be performed, by default None
        {None, "pos", "neg", "high", "low"}
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
        split = "high" or split = "low", in which case default is 0.3
    target : Optional[pd.DataFrame | str], optional
        Specify target for correlation. E.g. label column to generate only the \
        correlations between each feature and the label, by default None
    method : Literal['pearson', 'spearman', 'kendall'], optional
        method: {"pearson", "spearman", "kendall"}, by default "pearson"
        * pearson: measures linear relationships and requires normally distributed \
            and homoscedastic data.
        * spearman: ranked/ordinal correlation, measures monotonic relationships.
        * kendall: ranked/ordinal correlation, measures monotonic relationships. \
            Computationally more expensive but more robust in smaller dataets than \
            "spearman"
    colored : bool, optional
        If True the negative values in the correlation matrix are colored in red, by \
        default True

    Returns
    -------
    pd.DataFrame | pd.Styler
        If colored = True - corr: Pandas Styler object
        If colored = False - corr: Pandas DataFrame
    """
    # Validate Inputs
    _validate_input_range(threshold, "threshold", -1, 1)
    _validate_input_bool(colored, "colored")

    def color_negative_red(val: float) -> str:
        color = "#FF3344" if val < 0 else None
        return f"color: {color}"

    data = pd.DataFrame(data)

    _validate_input_num_data(data, "data")

    if isinstance(target, (str, list, pd.Series, np.ndarray)):
        target_data = []
        if isinstance(target, str):
            target_data = data[target]
            data = data.drop(target, axis=1)

        elif isinstance(target, (list, pd.Series, np.ndarray)):
            target_data = pd.Series(target)
            target = target_data.name

        corr = pd.DataFrame(
            data.corrwith(target_data, method=method, numeric_only=True),
        )
        corr = corr.sort_values(corr.columns[0], ascending=False)
        corr.columns = [target]

    else:
        corr = data.corr(method=method, numeric_only=True)

    corr = _corr_selector(corr, split=split, threshold=threshold)

    if colored:
        return corr.style.applymap(color_negative_red).format("{:.2f}", na_rep="-")
    return corr


def _corr_selector(
    corr: pd.Series | pd.DataFrame,
    split: Literal["pos", "neg", "high", "low"] | None = None,
    threshold: float = 0,
) -> pd.Series | pd.DataFrame:
    """Select the desired correlations using this utility function.

    Parameters
    ----------
    corr : pd.Series | pd.DataFrame
        pd.Series or pd.DataFrame of correlations
    split : Optional[str], optional
        Type of split performed, by default None
            * {None, "pos", "neg", "high", "low"}
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
        split = "high" or split = "low", in which case default is 0.3

    Returns
    -------
    pd.DataFrame
        List or matrix of (filtered) correlations
    """
    if split == "pos":
        corr = corr.where((corr >= threshold) & (corr > 0))
        print(
            'Displaying positive correlations. Specify a positive "threshold" to ' "limit the results further.",
        )
    elif split == "neg":
        corr = corr.where((corr <= threshold) & (corr < 0))
        print(
            'Displaying negative correlations. Specify a negative "threshold" to ' "limit the results further.",
        )
    elif split == "high":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) >= threshold)
        print(
            f"Displaying absolute correlations above the threshold ({threshold}). "
            'Specify a positive "threshold" to limit the results further.',
        )
    elif split == "low":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) <= threshold)
        print(
            f"Displaying absolute correlations below the threshold ({threshold}). "
            'Specify a positive "threshold" to limit the results further.',
        )

    return corr
