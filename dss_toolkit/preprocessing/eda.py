import pandas as pd


def categorical_and_numeric_columns(df):
    categorical_columns = df.select_dtypes("object").columns.tolist()
    numeric_columns = df.select_dtypes("number").columns.tolist()
    # TODO: Boolean variables are not included

    return categorical_columns, numeric_columns


def descriptive_statistics(df):

    categorical_columns, numeric_columns = categorical_and_numeric_columns(df)

    desc_stats_numeric = eda_numeric_cols(df, numeric_columns)
    desc_stats_categorical = eda_categorical_cols(df, categorical_columns)

    return {"numeric": desc_stats_numeric, "categorical": desc_stats_categorical}


def eda_numeric_cols(df, numeric_cols):
    """ Similar to describe() but with more quantiles"""
    # TODO: Add distribuition mini-chart
    l = []
    for c in numeric_cols:
        l.append(
            {
                "column_name": c,
                "mean_value": df[c].mean(),
                "count": df[c].count(),
                "missing": df[c].isna().astype(int).sum(),
                "missing_pct": df[c].isna().astype(int).sum() / df[c].shape[0],
                "min_value": df[c].min(),
                "q05": df[c].quantile(0.05),
                "q25": df[c].quantile(0.25),
                "q50": df[c].quantile(0.50),
                "q75": df[c].quantile(0.75),
                "q95": df[c].quantile(0.95),
                "max_value": df[c].max(),
            }
        )
    eda_df = pd.DataFrame(l)
    eda_df["dtypes"] = df[numeric_cols].dtypes.tolist()
    return eda_df


def eda_categorical_cols(df, categorical_cols):
    l = []
    for c in categorical_cols:
        u = df[c].value_counts()
        l.append(
            {
                "column_name": c,
                "count": df[c].count(),
                "missing": df[c].isna().astype(int).sum(),
                "missing_pct": df[c].isna().astype(int).sum() / df[c].shape[0],
                "n_unique": u.shape[0],
                "unique_vals": u.index.tolist(),
                "val_count": u.tolist(),
            }
        )
    eda_df = pd.DataFrame(l)
    eda_df["dtypes"] = df[categorical_cols].dtypes.tolist()
    return eda_df
