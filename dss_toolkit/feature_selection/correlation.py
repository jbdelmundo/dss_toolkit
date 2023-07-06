import numpy as np
from numpy import ndarray
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


def find_column_correlations(df):
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_pairs = pd.DataFrame()
    for column in upper.columns:
        corr = pd.DataFrame(upper[column].dropna())
        corr.columns = ["corr"]
        corr["index2"] = column
        corr.reset_index(inplace=True)
        corr_pairs = corr_pairs.append(corr)
    corr_pairs = corr_pairs[["index", "index2", "corr"]]
    corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
    return corr_pairs.sort_values("abs_corr", ascending=False)


class CorrelationFilter(BaseEstimator, SelectorMixin):
    def __init__(self, threshold=0.5, use_abs_corr=True, maximize_dropped=True):
        self.threshold = threshold
        self.use_abs_corr = use_abs_corr
        self.maximize_dropped = maximize_dropped
        self.dropped_columns_ = None
        self.orig_columns_ = None

    def fit(self, X, y=None):
        self.orig_columns_ = X.columns
        self.dropped_columns_ = find_correlated_columns_to_drop(
            X, threshold=self.threshold, use_abs_corr=self.use_abs_corr, maximize_dropped=self.maximize_dropped
        )
        # Remaining columns
        self.columns_ = [c for c in X.columns if c not in self.dropped_columns_]
        return self

    def transform(self, X, y=None):
        return X[self.columns_].copy()

    def get_feature_names_out(self, *args, **params):
        return self.columns_

    def get_support(self, indices: bool = False) -> ndarray:
        if indices:
            indices_ix = [ix for ix, c in enumerate(self.orig_columns_) if c not in self.dropped_columns_]
            return np.array(indices_ix)
        else:
            mask_values = [c not in self.dropped_columns_ for c in self.orig_columns_]
            return np.array(mask_values)

    def _get_support_mask(self, indices: bool = False):
        return self.get_support(indices=False)


def find_correlated_columns_to_drop(df, threshold=0.5, use_abs_corr=True, maximize_dropped=True):
    """
    Returns list of columns to drop based on correlation
    Parameters
    --------
    df: DataFrame
    threshold: float
        minimum threshold to be considered correlated pairs
    use_abs_corr: bool
        Whether to consider negative correlation
    maximize_dropped: bool
        if true, Drop first with the most correlated features.
        Suppose correlated pairs are (A,B), (A,C), (A,D), if set to True, this will drop B,C,D, if false, will drop A
    """

    # Compute corrleation matrix
    corr_matrix = df[df.select_dtypes("number").columns].corr()
    np.fill_diagonal(corr_matrix.values, np.nan)

    # Split into correlation pairs (feat_1, feat_2, correlation, abs_correlation)
    corr_pairs = corr_matrix.reset_index().melt(id_vars="index")
    corr_pairs.columns = ["feat_1", "feat_2", "correlation"]
    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
    corr_pairs.dropna(subset=["correlation"], inplace=True)

    # Drop features iteratively based on the related features
    dropped_features = []
    if use_abs_corr:
        drop_pairs = corr_pairs[corr_pairs.abs_correlation > threshold].copy()
    else:
        drop_pairs = corr_pairs[corr_pairs.correlation > threshold].copy()

    while drop_pairs.shape[0] > 0:
        feature_ranking = (
            drop_pairs.groupby("feat_1")
            .agg(
                related_feat_count=("feat_2", "count"),
                related_features=("feat_2", list),
            )
            .reset_index()
            .sort_values("related_feat_count", ascending=maximize_dropped)  # Sort based on number of related features
        ).reset_index(drop=True)
        #     display(feature_ranking)
        # Check for remaining features to drop
        if feature_ranking.shape[0] > 0:
            to_drop = feature_ranking["feat_1"][0]
            #         print("dropping", to_drop)
            dropped_features.append(to_drop)

            dropped_ix = drop_pairs[(drop_pairs.feat_1 == to_drop) | (drop_pairs.feat_2 == to_drop)].index
            drop_pairs.drop(dropped_ix, inplace=True)
    return dropped_features
