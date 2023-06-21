import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def find_VIF(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    return vif_data
