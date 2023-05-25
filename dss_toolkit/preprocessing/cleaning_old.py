import numpy as np
import pandas as pd
from dss_toolkit.preprocessing.data_validation import enforce_data_types

# raise Exception("Create the ff transformers: InvalidValueCleaner, DataTypeEnfocer, ")


class DataCleaner:
    """
    Class wrapper for the methods and supports dataframe-wide cleaning
    """

    def __init__(self, config):
        self.df = None

    def set_data(self, data):
        self.df = data

    def get_data(self, copy=False):
        if copy:
            return self.df.copy()
        else:
            return self.df

    def enforce_data_types(self, columns, data_types):
        return enforce_data_types(self.df, columns, data_types)

    def clean_whitespaces(self, columns=None):
        if columns is None:
            columns = self.df.columns
            print("Cleaning whitespaces for all columns")

        for c in columns:
            remove_whitespace(self.df, c, replace_nan=True, inplace=True)

    def replace_invalid_values(self, cleaning_rules):
        print("TODO: replace_invalid_values_df()-- verify rules")

        for column in cleaning_rules:
            rule = cleaning_rules[column]
            print(f"Applying cleaning rules for `{column}`: {rule}")
            replace_invalid_values(self.df, column=column, inplace=True, **rule)

    def replace_low_cardinality(self, columns, other_value="Others", cut_off_percentile=0.7):

        for column in columns:
            replace_low_cardinality(
                self.df, column=column, other_value=other_value, cut_off_percentile=cut_off_percentile, inplace=True,
            )

    def replace_values(self, columns, replacement_mappings):

        for (column, mapping) in zip(columns, replacement_mappings):

            self.df[column] = self.df[column].map(lambda x: mapping.get(x, x))

    def fit(self, data):
        pass

    def transform(self, data):
        pass


# Static methods here

"""
Cleaning Checklist
- Duplicate observations, unique attrs
- Range and possible values
- Outliers
- Typos, categories (N/A and NA to combine)
- Missing Values
- Validation based on business assumptions 

"""
def parse_data_dict_rules(
    data_dict_df,
    parameter_columns,
    undefined_str = "<<undefined_rule>>"
):
    valid_data_rules_df = data_dict_df[["column_name"] + parameter_columns].copy()
    valid_data_rules_df.fillna(undefined_str,inplace=True)  # Add invalid str
    
    # Aggregate columns with similar parameters
    agg_rules = valid_data_rules_df.groupby(parameter_columns).agg(
        columns_affected=("column_name",list)
    ).reset_index()
    
    # Convert rules into list
    agg_rules_list = agg_rules.to_dict(orient='records')
    
    # Remove empty parameters
    for rule in agg_rules_list:
        
        for parameter in parameter_columns:            
            # Remove rule parameter if parameter is undefined
            if rule[parameter] == undefined_str:
                rule.pop(parameter)
                continue
                
            if parameter == "valid_values":
                # Clean comma separated values
                rule['valid_values'] = rule['valid_values'].split(",")
                
    # Remove rules with zero parameters (only "columns_affected" left)
    agg_rules_list = [rule for rule in agg_rules_list if len(rule.keys()) > 1 ]
    
    return agg_rules_list

class OutlierDetection:
    def __init__(
        self, how="iqr", zscore_threhsold=3, treatment="cap", cap_percentile=(0.10, 0.90), outlier_replacement=0,
    ):
        if how not in ["iqr", "zscore"]:
            raise Exception("Supported Methods are 'iqr' or 'zscore'")

        if treatment not in ["clear", "drop", "cap", "mean", "median", "constant"]:
            raise Exception("Supported Methods are 'clear','drop','cap','mean','median' ")

        self.how = how
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
        if self.how == "iqr":
            low_outlier = df < self.iqr_lower_limit
            high_outlier = df > self.iqr_upper_limit

        elif self.how == "zscore":
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

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)





class ValuesValidator:
    def __init__(self,
        valid_values=[], 
        min_value=None, 
        max_value=None, 
        replacement=None, 
        drop_invalid=False
    ):

        pass

    def fit(self, data):
        return self

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self):
        pass


def replace_invalid_values(
    df, column, drop_invalid=False, possible_values=[], min_value=None, max_value=None, replacement=None, inplace=False
):
    """
    Validate values

    Parameters
    -----------------
    df : Dataframe
    column : str
        Name of the column to clean
    drop_invalid : bool (default: False)
        If set to True, drops rows with invalid values, otherwise values are replaced with the
        `replacement` parameter
    possible_values: list
        List of possible values for categorical variable
    min_value:  numeric (default: None)
        Minimum value for numeric variable
    max_value: numeric (default: None)
        Maximum value for numeric variable
    replacement: str or numeric (default: None)
        If not set to `None`, replaces invalid values with this value
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

    # Handle all invalid records
    if inplace:
        if drop_invalid:
            df.drop(invalid_rows_all.index, inplace=True)
        else:
            df.loc[invalid_rows_all.index, column] = replacement
    else:
        df_new = df.copy()
        if drop_invalid:
            df_new.drop(invalid_rows_all.index, inplace=True)
        else:
            df_new.loc[invalid_rows_all.index, column] = replacement
        return df_new


def remove_whitespace(df_p, column, replace_nan=True, inplace=False):
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


def replace_iqr_outliers(
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
    if not inplace:
        df = df.copy()

    df.loc[df[column] < lower_limit, column] = min_val
    df.loc[df[column] > upper_limit, column] = max_val

    return df


def replace_low_cardinality(df, column, cut_off_percentile=0.3, other_value="Others", inplace=False):
    """
    Replaces categorical values with low cardinality to limit possible values

    Parameters
    -----------
    df : Dataframe
    column: str
        Name of the column to clean
    cut_off_percentile : float (default: 0.3)
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
