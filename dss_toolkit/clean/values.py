import numpy as np

import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin


class InvalidValuesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, rules=[]):
        """
        # Tuple of (column, args, replacement)
        where `args` {"min_value":min_val, "max_value":max_val, "possible_values":[]}
        """
        # TODO add validation for rules
        self.rules = rules
        for rule_ix, rule in enumerate(rules):
            self.__validate_invalidvaluescleaner_rule(rule, rule_ix)
        self.columns_ = None

    def add_rule(self, rule):
        self.__validate_invalidvaluescleaner_rule(rule, rule_ix=0)
        self.rules.append(rule)

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        self.columns_ = X.columns
        df = X.copy()  # Create a copy to modify
        for column, kwargs, replacement in self.rules:
            clean_invalid_values(df, column=column, inplace=True, replacement=replacement, **kwargs)
        return df

    def __validate_invalidvaluescleaner_rule(self, rule, rule_ix=None):
        if len(rule) != 3:
            msg = f"Rule at index {rule_ix} should be a tuple of length 3: (column, args, replacement)"
            raise ValueError(msg)

        column, args, replacement = rule

        if len(args.keys()) == 0:
            msg = 'Atleast 1 argument required: ["min_value", "max_value","valid_values"]"'
            raise ValueError(msg)

        # Validate arguments
        for arg in list(args.keys()):
            if arg not in ["min_value", "max_value", "valid_values"]:
                msg = f'Argument {arg} not in ["min_value", "max_value","valid_values"]"'
                raise ValueError(msg)

            if arg == "valid_values" and type(args["valid_values"]) != list:
                msg = "Argument `possible_values` should be a list"
                raise ValueError(msg)

            if arg in ["min_value", "max_value"] and type(args[arg]) not in [int, float]:
                msg = f"Argument `{arg}` should be a number"
                raise ValueError(msg)

    def get_feature_names_out(self, *args, **params):
        return self.columns_


def clean_invalid_values(
    df, column, mode="replace", valid_values=[], min_value=None, max_value=None, replacement=None, inplace=False
):
    """
    Validate values

    Parameters
    -----------------
    df : Dataframe
    column : str
        Name of the column to clean
    mode : str ("replace" or "drop") Default: replace
        replace :  repalce values with the parameter `replacement` (default: `None`)
        drop: drop rows with invalid values
    possible_values: list
        List of possible values for categorical variable
    min_value:  numeric (default: None)
        Minimum value for numeric variable
    max_value: numeric (default: None)
        Maximum value for numeric variable
    replacement: str or numeric (default: None)
        If `mode="replace"`, replaces invalid values with this value
    inplace: boolean (default: False)
        returns a copy if set to `True`

    Returns
    --------
    Dataframe
    """
    invalid_rows_all = pd.DataFrame()
    if column not in df.columns:
        msg = f"Column `{column}` from the rules given is not found in the dataframe"
        raise ValueError(msg)

    # Indentify invalid rows
    # For categorical column
    if len(valid_values) > 0 and type(valid_values) == list:
        invalid_rows = df[~df[column].isna() & ~df[column].isin(valid_values)]
        invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

    # For numeric column
    if min_value is not None:
        invalid_rows = df[df[column] < min_value]
        invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

    if max_value is not None:
        invalid_rows = df[df[column] > max_value]
        invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

    if inplace:
        if mode == "drop" and replacement is None:
            df.drop(invalid_rows_all.index, inplace=True)
        elif replacement is not None:
            df.loc[invalid_rows_all.index, column] = replacement
    else:
        df_new = df.copy()
        if mode == "drop" and replacement is None:
            df_new.drop(invalid_rows_all.index, inplace=True)
        elif replacement is not None:
            df_new.loc[invalid_rows_all.index, column] = replacement
        return df_new


def str_strip_whitespace(df_p, column, replace_nan=True, inplace=False):
    """
    Removes whitespace from a string column

    Parameters
    -----------------
    df_p : Dataframe
    column : str
        Name of the column to clean
    replace_nan: boolean (default: True)
        Replaces empty strings ("") with `np.nan`
    inplace: boolean (default: False)
        Returns a copy if set to `True`

    Returns
    --------
    Dataframe
    """
    if not inplace:
        df = df_p.copy()
    else:
        df = df_p

    df.loc[:, column] = df[column].str.strip()
    if replace_nan:
        df.loc[:, column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    return df


class HighCardinlityBinning(TransformerMixin, BaseEstimator):
    def __init__(self, top_n=0.9, max_freq=10, other_value="Others", exclude=[]):
        self.top_n = top_n
        self.max_freq = max_freq
        self.other_value = other_value
        self.exclude = exclude
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        self.columns_ = X.columns
        data_cleaned = replace_low_cardinality(
            X, max_freq=self.max_freq, top_n=self.top_n, other_value=self.other_value, exclude=self.exclude
        )
        return data_cleaned

    def get_feature_names_out(self, *args, **params):
        return self.columns_


def replace_low_cardinality(data, max_freq=10, top_n=10, other_value="Other", exclude=[]):
    """
     Replaces categorical values with low cardinality to limit possible values

    Parameters
    -----------
    data : pd.Dataframe
    max_freq: int
        number of unique values per column to be considered high cardinality
    top_n : float (default: 0.9)
        When type of `int`, selects top `top_n` values based on frequency.
        When type of `float`, selects top `top_n` perceent values based on cumulative frequency. Default value is 0.9,
        (selects values on top 90% common values, the lower 10% are renamed as `other_value` )
    other_value: str (default: "Others")
        Replacement value for the low cardinality
    exclude: list of columns to exclude
    """
    data = data.copy()  # Preserve original
    high_cardinality = __find_high_cardinality_columns(data, max_freq, exclude)
    data[high_cardinality] = data[high_cardinality].apply(
        __replace_low_cardinality_series, top_n=top_n, other_value=other_value
    )
    return data


def __find_high_cardinality_columns(df, max_freq, exclude=[]):
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

    high_cardinal = []
    for c in categorical_columns:
        unique_vals = df[c].dropna().unique().size
        if unique_vals > max_freq and c not in exclude:
            high_cardinal.append(c)
    return high_cardinal


def __replace_low_cardinality_series(series, top_n=0.9, other_value="Others", inplace=False):
    """
    Replaces categorical values with low cardinality to limit possible values

    Parameters
    -----------
    series : pd.Series
    column: str
        Name of the column to clean
    top_n : float (default: 0.9)
        When type of `int`, selects top `top_n` values based on frequency.
        When type of `float`, selects top `top_n` perceent values based on cumulative frequency. Default value is 0.9,
        (selects values on top 90% common values, the lower 10% are renamed as `other_value` )
    other_value: str (default: "Others")
        Replacement value for the low cardinality

    inplace: boolean (default: False)
        Returns a copy if set to `True`

    """
    if type(top_n) == float:
        top_n_pct = top_n
        top_n = None

    freq_pct = series.value_counts(normalize=True)
    cum_freq_pct = freq_pct.cumsum()

    if top_n:
        #  top N items
        top_values = freq_pct[:top_n].index.tolist()
    elif top_n_pct:
        # top n%
        top_values = cum_freq_pct[cum_freq_pct <= top_n_pct].index.tolist()
    else:
        return series

    if not inplace:
        series = series.copy()
    series.loc[~series.isin(top_values)] = other_value

    return series
