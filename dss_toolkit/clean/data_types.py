from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from dss_toolkit.base.input_validation import _validate_input_bool, _validate_input_range


class BinaryNumericDtypeCleaner(TransformerMixin, BaseEstimator):
    """
    Transformer Wrapper for scikit-learn
    """

    def __init__(self, na_fill="", exclude=[]):
        self.exclude = exclude
        self.na_fill = na_fill

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        data_cleaned = fix_binary_numeric_dtype(data, na_fill=self.na_fill, exclude=self.exclude)
        return data_cleaned


def fix_binary_numeric_dtype(data, na_fill="", exclude=[]) -> pd.DataFrame:
    data = pd.DataFrame(data).copy()

    unique_counts = data[data.select_dtypes("number").columns].nunique()
    binary_numeric = unique_counts[unique_counts == 2].index.tolist()
    binary_numeric = [c for c in binary_numeric if c not in exclude]

    data[binary_numeric] = data[binary_numeric].apply(lambda s: s.fillna(na_fill).astype(str))
    # Replace Empty Strings with np.nan
    #     data[binary_numeric] = data[binary_numeric].replace(r"^\s*$", np.nan, regex=True)

    data[binary_numeric] = data[binary_numeric].replace("", np.nan)  # no need for regex matching
    return data


def convert_datatypes(
    data: pd.DataFrame,
    category: bool = True,
    cat_threshold: float = 0.05,
    cat_exclude: list[str | int] | None = None,
) -> pd.DataFrame:
    """Convert columns to best possible dtypes using dtypes supporting pd.NA.

    Temporarily not converting to integers due to an issue in pandas. This is expected \
        to be fixed in pandas 1.1. See https://github.com/pandas-dev/pandas/issues/33803

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    category : bool, optional
        Change dtypes of columns with dtype "object" to "category". Set threshold \
        using cat_threshold or exclude columns using cat_exclude, by default True
    cat_threshold : float, optional
        Ratio of unique values below which categories are inferred and column dtype is \
        changed to categorical, by default 0.05
    cat_exclude : Optional[list[str | int]], optional
        List of columns to exclude from categorical conversion, by default None

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with converted Datatypes
    """
    # Validate Inputs
    _validate_input_bool(category, "Category")
    _validate_input_range(cat_threshold, "cat_threshold", 0, 1)

    cat_exclude = [] if cat_exclude is None else cat_exclude.copy()

    data = pd.DataFrame(data).copy()
    for col in data.columns:
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if category and unique_vals_ratio < cat_threshold and col not in cat_exclude and data[col].dtype == "object":
            data[col] = data[col].astype("category")

        data[col] = data[col].convert_dtypes(
            convert_integer=False,
            convert_floating=False,
        )

    data = _optimize_ints(data)
    data = _optimize_floats(data)

    return data


class OptimizeNumbers(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        self.columns_ = X.columns
        return optimize_numbers(X)

    def get_feature_names_out(self, *args, **params):
        return self.columns_


def optimize_numbers(data):
    data = pd.DataFrame(data).copy()
    data = _optimize_ints(data)
    data = _optimize_floats(data)
    return data


def _optimize_ints(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(data).copy()  # noqa: PD901
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def _optimize_floats(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(data).copy()
    floats = data.select_dtypes(include=["float64"]).columns.tolist()
    data[floats] = data[floats].apply(pd.to_numeric, downcast="float")
    return data
