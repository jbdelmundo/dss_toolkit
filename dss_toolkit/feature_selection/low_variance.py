from sklearn.feature_selection import VarianceThreshold


# drop single_valued columns
def drop_single_valued_columns(df):
    unique_counts = df[df.select_dtypes("number").columns].nunique()
    single_value_cols = unique_counts[unique_counts == 1].index.tolist()
    return df.drop(columns=single_value_cols).copy()


def get_low_variance_columns(df, **kwargs):
    low_variance_threshold = kwargs.get("low_variance_threshold", 0)

    variance_selector = VarianceThreshold(threshold=low_variance_threshold)
    variance_selector.fit(df)

    low_variance_columns = df.columns[~variance_selector.get_support()].tolist()
    print("Low Variance Columns:", low_variance_columns)
    return low_variance_columns
