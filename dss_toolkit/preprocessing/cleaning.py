import numpy as np
import pandas as pd


class InvalidValuesCleaner:
    def __init__(self):
        self.rules = []

    def add_rule(
        self, column, valid_values=None, min_value=None, max_value=None, replacement=np.nan,
    ):
        # TODO: Add `max_length` for string
        # TODO: Update to transformer to cover multiple columns with single rule
        if valid_values is not None and len(valid_values) == 0:
            raise Exception("Parameter `valid_values` should not be empty")

        self.rules.append(
            {
                "column_name": column,
                "valid_values": valid_values,
                "min_value": min_value,
                "max_value": max_value,
                "replacement": replacement,
            }
        )

    def fit(self, *args, **kwargs):
        pass  # for compatibilty

    def fit_transform(self, df):
        return self.transform(df)  # Same as transform()

    def transform(self, df):
        if len(self.rules) == 0:
            print("No rules found. Add using `add_numeric_rule()` or `add_categorical_rule()`")
            return

        df_copy = df.copy()  # Create a copy
        # Iterate over the rules
        for rule in self.rules:
            self._check_column(
                df_copy,
                rule["column_name"],
                possible_values=rule["valid_values"],
                min_value=rule["min_value"],
                max_value=rule["max_value"],
                replacement=rule["replacement"],
                inplace=True,
            )

        return df_copy

    def _check_column(
        self,
        df,
        column,
        drop_invalid=False,
        possible_values=[],
        min_value=None,
        max_value=None,
        replacement=None,
        inplace=False,
    ):
        invalid_rows_all = pd.DataFrame()

        # Indentify invalid rows
        # For categorical column
        if possible_values is not None and len(possible_values) > 0 and type(possible_values) == list:
            invalid_rows = df[~df[column].isna() & ~df[column].isin(possible_values)]
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

        # For numeric column
        # TODO: for invalid, add option to 'cap' instead of tagging as invalid
        if min_value is not None:
            invalid_rows = df[df[column] < min_value]
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

        if max_value is not None:
            invalid_rows = df[df[column] > max_value]
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_rows], axis=0)

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
