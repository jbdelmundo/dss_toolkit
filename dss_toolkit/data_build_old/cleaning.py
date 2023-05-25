import numpy as np
import pandas as pd


def qa_values(
    df, column, mode="replace", possible_values=[], min_value=None, max_value=None, replacement=None, inplace=False
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

    # Indentify invalid rows
    # For categorical column
    if len(possible_values) > 0 and type(possible_values) == list:
        invalid_rows = df[~df[column].isna() & ~df[column].isin(possible_values)]
        invalid_rows_all = invalid_rows_all.append(invalid_rows)

    # For numeric column
    if min_value is not None:
        invalid_rows = df[df[column] < min_value]
        invalid_rows_all = invalid_rows_all.append(invalid_rows)

    if max_value is not None:
        invalid_rows = df[df[column] > max_value]
        invalid_rows_all = invalid_rows_all.append(invalid_rows)

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


def qa_strip_whitespace(df_p, column, replace_nan=True, inplace=False):
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


def replace_iqr_outlier(
    df, column, quantile_range=(0.25, 0.75), replacement_range=(0.05, 0.95), mode="replace", inplace=False
):
    """
    Replace outliers with 5th and 95th percentile values
    Outliers are detected beyond 1.5 * interquartile range

    Parameters
    -----------
    df : Dataframe
    column: str
        Name of the column to clean
    quantile_range: tuple (double, double)
        Identifies the IQR range. Default is `(0.25, 0.75)`
    replacement_range: tuple (double, double)
        identifies the replacement value for outliers
        Default is 5th percentile and 95th percentile (0.05, 0.95)
    mode: str (default: "replace")
        "replace": replaces outliers with 5th and 95th percentile values (or defined by `replacement_range`)
        "cap": replaces outliers with 1.5 IQR 
        None: replaces outliers with `np.nan`
    inplace: boolean (default: False)
        Returns a copy if set to `True`

    Returns
    --------
    Dataframe 

    """
    q1 = df[column].quantile(quantile_range[0])
    q3 = df[column].quantile(quantile_range[1])
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    if mode == "cap":
        min_val = lower_limit
        max_val = upper_limit
    elif mode == "replace":
        min_val = df[column].quantile(replacement_range[0])
        max_val = df[column].quantile(replacement_range[1])
    else:
        # Empty
        min_val = np.nan
        max_val = np.nan

    # Treat Outliers
    if inplace:
        df.loc[df[column] < lower_limit, column] = min_val
        df.loc[df[column] > upper_limit, column] = max_val
    else:
        df = df.copy()
        df.loc[df[column] < lower_limit, column] = min_val
        df.loc[df[column] > upper_limit, column] = max_val

    return df


def replace_low_cardinality(df, column, cut_off_percentile=0.7, other_value="Others", inplace=False):
    """
    Replaces categorical values with low cardinality to limit possible values

    Parameters
    -----------
    df : Dataframe
    column: str
        Name of the column to clean
    cut_off_percentile : float (default: 0.7)
        Only include values whose occurences is above this percentile. By default, includes only with occurences above 70th percentile.
        Occurrences with the bottom 30% will not be included
    other_value: str (default: "Others")
        Replacement value for the low cardinality
    
    inplace: boolean (default: False)
        Returns a copy if set to `True`
    
    """

    freq = df[column].value_counts(True)
    cut_off = freq.quantile(cut_off_percentile)
    less_freq_values = freq[freq < cut_off].index.tolist()
    if not inplace:
        df = df.copy()
    df.loc[df[column].isin(less_freq_values), column] = other_value

    return df


def scale_division(
    df, column, divisor_column=None, divisor_series=None, divisor_value=None, bias=0.000001,
):

    if divisor_column is not None:
        df.loc[:, column] = df[column] / (df[divisor_column] + bias)  # prevent div by zero
    elif divisor_series is not None:
        df.loc[:, column] = df[column] / (divisor_series + bias)
    else:
        df.loc[:, column] = df[column] / divisor_value
