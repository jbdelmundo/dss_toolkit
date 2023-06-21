import pandas as pd


def number_analyzer(df, numeric_columns=None):
    """Similar to describe() but with more quantiles"""

    if numeric_columns is None:
        numeric_columns = df.select_dtypes("number").columns.tolist()

    # TODO: Add distribuition mini-chart
    column_stats = []
    for c in numeric_columns:
        column_stats.append(
            {
                "column_name": c,
                "mean_value": df[c].mean(),
                "std_dev": df[c].std(),
                "n_unique": df[c].value_counts().size,
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
    eda_df = pd.DataFrame(
        column_stats,
        columns=[
            "column_name",
            "mean_value",
            "std_dev",
            "n_unique",
            "count",
            "missing",
            "missing_pct",
            "min_value",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "max_value",
        ],
    )
    eda_df["dtypes"] = df[numeric_columns].dtypes.tolist()
    return eda_df


def categorical_analyzer(df, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    column_stats = []
    for c in categorical_cols:
        u = df[c].value_counts()
        column_stats.append(
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
    eda_df = pd.DataFrame(
        column_stats,
        columns=["column_name", "count", "missing", "missing_pct", "n_unique", "unique_vals", "val_count"],
    )
    eda_df["dtypes"] = df[categorical_cols].dtypes.tolist()
    return eda_df
