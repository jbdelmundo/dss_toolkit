from __future__ import annotations
import pandas as pd
from typing import Literal, TypedDict

from sklearn.base import BaseEstimator, TransformerMixin
from dss_toolkit.base.input_validation import _validate_input_range
from dss_toolkit.analysis.correlation import corr_mat


def mv_col_handling(
    data: pd.DataFrame,
    target: str | pd.Series | list[str] | None = None,
    mv_threshold: float = 0.1,
    corr_thresh_features: float = 0.5,
    corr_thresh_target: float = 0.3,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str], list[str]]:
    """Convert columns with a high ratio of missing values into binary features.

    Eventually drops them based on their correlation with other features and the \
    target variable.

    This function follows a three step process:
    - 1) Identify features with a high ratio of missing values (above 'mv_threshold').
    - 2) Identify high correlations of these features among themselves and with \
        other features in the dataset (above 'corr_thresh_features').
    - 3) Features with high ratio of missing values and high correlation among each \
        other are dropped unless they correlate reasonably well with the target \
        variable (above 'corr_thresh_target').

    Note: If no target is provided, the process exits after step two and drops columns \
    identified up to this point.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    target : Optional[str | pd.Series | list]], optional
        Specify target for correlation. I.e. label column to generate only the \
        correlations between each feature and the label, by default None
    mv_threshold : float, optional
        Value between 0 <= threshold <= 1. Features with a missing-value-ratio larger \
        than mv_threshold are candidates for dropping and undergo further analysis, by \
        default 0.1
    corr_thresh_features : float, optional
        Value between 0 <= threshold <= 1. Maximum correlation a previously identified \
        features (with a high mv-ratio) is allowed to have with another feature. If \
        this threshold is overstepped, the feature undergoes further analysis, by \
        default 0.5
    corr_thresh_target : float, optional
        Value between 0 <= threshold <= 1. Minimum required correlation of a remaining \
        feature (i.e. feature with a high mv-ratio and high correlation to another \
        existing feature) with the target. If this threshold is not met the feature is \
        ultimately dropped, by default 0.3
    return_details : bool, optional
        Provdies flexibility to return intermediary results, by default False

    Returns
    -------
    pd.DataFrame
        Updated Pandas DataFrame

    optional:
    cols_mv: Columns with missing values included in the analysis
    drop_cols: List of dropped columns
    """
    # Validate Inputs
    _validate_input_range(mv_threshold, "mv_threshold", 0, 1)
    _validate_input_range(corr_thresh_features, "corr_thresh_features", 0, 1)
    _validate_input_range(corr_thresh_target, "corr_thresh_target", 0, 1)

    data = pd.DataFrame(data).copy()
    data_local = data.copy()
    mv_ratios = _missing_vals(data_local)["mv_cols_ratio"]
    cols_mv = mv_ratios[mv_ratios > mv_threshold].index.tolist()
    data_local[cols_mv] = data_local[cols_mv].applymap(lambda x: x if pd.isna(x) else 1).fillna(0)

    high_corr_features = []
    data_temp = data_local.copy()
    for col in cols_mv:
        corrmat = corr_mat(data_temp, colored=False)
        if abs(corrmat[col]).nlargest(2)[1] > corr_thresh_features:
            high_corr_features.append(col)
            data_temp = data_temp.drop(columns=[col])

    drop_cols = []
    if target is None:
        data = data.drop(columns=high_corr_features)
    else:
        corrs = corr_mat(data_local, target=target, colored=False).loc[high_corr_features]
        drop_cols = corrs.loc[abs(corrs.iloc[:, 0]) < corr_thresh_target].index.tolist()
        data = data.drop(columns=drop_cols)

    return (data, cols_mv, drop_cols) if return_details else data


class MissingValuesDropper(BaseEstimator, TransformerMixin):
    """
    Drop completely empty columns and rows by default.

    Optionally provides flexibility to loosen restrictions to drop additional \
    non-empty columns and rows based on the fraction of NA-values.
    Dropping of columns is done before computation of row-wise missing values
    """

    def __init__(self, col_threshold: float = 1, row_threshold: float = 1, col_exclude: list[str] | None = None):
        self.col_threshold = col_threshold
        self.row_threshold = row_threshold
        self.col_exclude = col_exclude

        self.columns_ = None

    def fit(self, X):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        self.columns_ = X.columns
        return drop_missing(
            X,
            drop_threshold_cols=self.col_threshold,
            drop_threshold_rows=self.row_threshold,
            col_exclude=self.col_exclude,
        )

    def get_feature_names_out(self, *args, **params):
        return self.columns_


def drop_missing(
    data: pd.DataFrame,
    drop_threshold_cols: float = 1,
    drop_threshold_rows: float = 1,
    col_exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Drop completely empty columns and rows by default.

    Optionally provides flexibility to loosen restrictions to drop additional \
    non-empty columns and rows based on the fraction of NA-values.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    drop_threshold_cols : float, optional
        Drop columns with NA-ratio equal to or above the specified threshold, by \
        default 1
    drop_threshold_rows : float, optional
        Drop rows with NA-ratio equal to or above the specified threshold, by default 1
    col_exclude : Optional[list[str]], optional
        Specify a list of columns to exclude from dropping. The excluded columns do \
        not affect the drop thresholds, by default None

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame without any empty columns or rows

    Notes
    -----
    Columns are dropped first
    """
    # Validate Inputs
    _validate_input_range(drop_threshold_cols, "drop_threshold_cols", 0, 1)
    _validate_input_range(drop_threshold_rows, "drop_threshold_rows", 0, 1)

    col_exclude = [] if col_exclude is None else col_exclude.copy()
    data_exclude = data[col_exclude]

    data = pd.DataFrame(data).copy()

    data_dropped = data.drop(columns=col_exclude, errors="ignore")
    data_dropped = data_dropped.drop(
        columns=data_dropped.loc[
            :,
            _missing_vals(data)["mv_cols_ratio"] > drop_threshold_cols,
        ].columns,
    ).dropna(axis=1, how="all")

    data = pd.concat([data_dropped, data_exclude], axis=1)

    return data.drop(
        index=data.loc[
            _missing_vals(data)["mv_rows_ratio"] > drop_threshold_rows,
            :,
        ].index,
    ).dropna(axis=0, how="all")


class MVResult(TypedDict):
    """TypedDict for the return value of _missing_vals."""

    mv_total: int
    mv_rows: int
    mv_cols: int
    mv_rows_ratio: float
    mv_cols_ratio: float


def _missing_vals(data: pd.DataFrame) -> MVResult:
    """Give metrics of missing values in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame

    Returns
    -------
    Dict[str, float]
        mv_total: float, number of missing values in the entire dataset
        mv_rows: float, number of missing values in each row
        mv_cols: float, number of missing values in each column
        mv_rows_ratio: float, ratio of missing values for each row
        mv_cols_ratio: float, ratio of missing values for each column
    """
    data = pd.DataFrame(data).copy()
    mv_total: int = data.isna().sum().sum()
    mv_rows: int = data.isna().sum(axis=1)
    mv_cols: int = data.isna().sum(axis=0)
    mv_rows_ratio: float = mv_rows / data.shape[1]
    mv_cols_ratio: float = mv_cols / data.shape[0]

    return {
        "mv_total": mv_total,
        "mv_rows": mv_rows,
        "mv_cols": mv_cols,
        "mv_rows_ratio": mv_rows_ratio,
        "mv_cols_ratio": mv_cols_ratio,
    }
