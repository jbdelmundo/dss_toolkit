import numpy as np
import pandas as pd


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
