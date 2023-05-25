import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor


def appy_boruta(df, numeric_features, categorical_features, target):

    # Onehot categorical, keep numeric
    X = encode_X(df, numeric_features, categorical_features)
    y = df[target]

    forest = RandomForestRegressor(n_jobs=-1, n_estimators=100, max_depth=100, random_state=42)
    forest.fit(X, y)
    feat_selector = BorutaPy(forest, n_estimators="auto", verbose=0, random_state=1)
    #     return X
    feat_selector.fit(X.values, y.values)

    results = pd.DataFrame(
        {"feature": X.columns, "support": feat_selector.support_, "ranking": feat_selector.ranking_}
    ).sort_values(["ranking", "support"])

    return results


def get_low_variance_columns(df, **kwargs):
    low_variance_threshold = kwargs.get("low_variance_threshold", 0)

    variance_selector = VarianceThreshold(threshold=low_variance_threshold)
    variance_selector.fit(df)

    low_variance_columns = df.columns[~variance_selector.get_support()].tolist()
    print("Low Variance Columns:", low_variance_columns)
    return low_variance_columns


def find_column_correlations(df):
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

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


def find_VIF(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    return vif_data


def encode_X(df, numeric_features, categorical_features):
    X_num = df[numeric_features]
    X_cat_demogs = pd.get_dummies(df[["marital_status", "gender"]], drop_first=True)
    X_cat_product = pd.get_dummies(df[["first_product_bef_first_auto", "product_bef_first_auto"]], drop_first=False)
    X = pd.concat([X_num, X_cat_demogs, X_cat_product], axis=1).reset_index().drop(columns="index")
    return X
