import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


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


def encode_X(df, numeric_features, categorical_features):
    X_num = df[numeric_features]
    X_cat_demogs = pd.get_dummies(df[["marital_status", "gender"]], drop_first=True)
    X_cat_product = pd.get_dummies(df[["first_product_bef_first_auto", "product_bef_first_auto"]], drop_first=False)
    X = pd.concat([X_num, X_cat_demogs, X_cat_product], axis=1).reset_index().drop(columns="index")
    return X
