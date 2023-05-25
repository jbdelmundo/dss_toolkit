import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class OutlierDetection(TransformerMixin, BaseEstimator):
    def __init__(
        self, detection="iqr", zscore_threhsold=3, treatment="cap", cap_percentile=(0.10, 0.90), outlier_replacement=0,
    ):
        if detection not in ["iqr", "zscore"]:
            raise Exception("Supported Methods are 'iqr' or 'zscore'")

        if treatment not in ["clear", "drop", "cap", "mean", "median", "constant"]:
            raise Exception("Supported Methods are 'clear','drop','cap','mean','median' ")

        self.detection = detection
        self.zscore_threhsold = zscore_threhsold  # num of standard deviations from the mean to consider
        self.treatment = treatment
        self.cap_percentile = cap_percentile
        self.outlier_replacement = outlier_replacement

    def fit(self, df):
        df = pd.DataFrame(df)

        self.mean = df.mean(axis=0)
        self.median = df.quantile(0.50)
        self.std = df.std(axis=0)
        self.q1 = df.quantile(0.25)
        self.q3 = df.quantile(0.75)

        self.q_low_cap = df.quantile(self.cap_percentile[0])  # 10th percentile default
        self.q_high_cap = df.quantile(self.cap_percentile[1])  # 90th percentile default

        iqr = self.q3 - self.q1
        self.iqr_lower_limit = self.q1 - (1.5 * iqr)
        self.iqr_upper_limit = self.q3 + (1.5 * iqr)

    def transform(self, df):
        df = pd.DataFrame(df)

        # Identify Outliers
        if self.detection == "iqr":
            low_outlier = df < self.iqr_lower_limit
            high_outlier = df > self.iqr_upper_limit

        elif self.detection == "zscore":
            z = (df - self.mean) / self.std
            low_outlier = z < (-1 * self.zscore_threhsold)
            high_outlier = z > self.zscore_threhsold
        else:
            return df

        # Treat outliers
        if self.treatment == "drop":

            all_outlier = low_outlier | high_outlier
            outlier_index = all_outlier[all_outlier.sum(axis=1) > 0].index
            df.drop(outlier_index, inplace=True)  # Drops rows with atleast 1 outlier

        elif self.treatment in ["clear", "mean", "median", "constant"]:
            all_outlier = low_outlier | high_outlier

            for col in df.columns:

                # Identify proper replacement
                if self.treatment == "clear":
                    replacement = None
                elif self.treatment == "mean":
                    replacement = self.mean[col]
                elif self.treatment == "median":
                    replacement = self.median[col]
                else:
                    replacement = self.outlier_replacement  # default

                # Replace outliers per column
                outlier_map = all_outlier[col]
                df.loc[outlier_map, col] = replacement

        elif self.treatment == "cap":

            for col in df.columns:

                # Replace small outliers per column
                low_outlier_map = low_outlier[col]
                df.loc[low_outlier_map, col] = self.q_low_cap[col]

                # Replace large outliers per column
                high_outlier_map = high_outlier[col]
                df.loc[high_outlier_map, col] = self.q_high_cap[col]

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def set_output(self, *, transform=None):
        pass  # super().set_output(transform=transform)
