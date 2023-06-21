from __future__ import annotations
import re
import numpy as np
import pandas as pd


def enforce_data_types(df, column_names, column_data_types):
    if len(column_data_types) != len(column_names):
        raise Exception("Column names and data dtypes have different lengths")

    mapping = {
        "str": str,
        "string": str,
        "int": int,
        "float": float,
        "decimal": float,
        "float32": np.float32,
        "float64": np.float64,
    }

    for c, t_str in zip(column_names, column_data_types):
        t = mapping.get(t_str)
        if t is None:
            raise Exception(f"Data type {t_str} not recognized. Known data types are: {mapping.keys()}")

        try:
            df[c] = df[c].astype(t)
        except ValueError as e:
            print(f"Cannot cast `{c}` to {t_str}, casting to float instead. Error:", e)
            df[c] = df[c].astype(float)


def clean_column_names(data: pd.DataFrame, hints: bool = True) -> pd.DataFrame:
    """Clean the column names of the provided Pandas Dataframe.

    Optionally provides hints on duplicate and long column names.

    Parameters
    ----------
    data : pd.DataFrame
        Original Dataframe with columns to be cleaned
    hints : bool, optional
        Print out hints on column name duplication and colum name length, by default \
        True

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with cleaned column names
    """
    # _validate_input_bool(hints, "hints")

    # Handle CamelCase
    for i, col in enumerate(data.columns):
        matches = re.findall(re.compile("[a-z][A-Z]"), col)
        column = col
        for match in matches:
            column = column.replace(match, f"{match[0]}_{match[1]}")
            data = data.rename(columns={data.columns[i]: column})

    data.columns = (
        data.columns.str.replace("\n", "_", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "_", regex=False)
        .str.replace("'", "_", regex=False)
        .str.replace('"', "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[!?:;/]", "_", regex=True)
        .str.replace("+", "_plus_", regex=False)
        .str.replace("*", "_times_", regex=False)
        .str.replace("<", "_smaller", regex=False)
        .str.replace(">", "_larger_", regex=False)
        .str.replace("=", "_equal_", regex=False)
        .str.replace("ä", "ae", regex=False)
        .str.replace("ö", "oe", regex=False)
        .str.replace("ü", "ue", regex=False)
        .str.replace("ß", "ss", regex=False)
        .str.replace("%", "_percent_", regex=False)
        .str.replace("$", "_dollar_", regex=False)
        .str.replace("€", "_euro_", regex=False)
        .str.replace("@", "_at_", regex=False)
        .str.replace("#", "_hash_", regex=False)
        .str.replace("&", "_and_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )

    if dupl_idx := [i for i, x in enumerate(data.columns.duplicated()) if x]:
        dupl_before = data.columns[dupl_idx].tolist()
        data.columns = [
            col if col not in data.columns[:i] else f"{col}_{str(i)}" for i, col in enumerate(data.columns)
        ]
        print_statement = ""
        if hints:
            # add string to print statement
            print_statement = (
                f"Duplicate column names detected! Columns with index {dupl_idx} and "
                f"names {dupl_before} have been renamed to "
                f"{data.columns[dupl_idx].tolist()}.",
            )

            if long_col_names := [x for x in data.columns if len(x) > 25]:  # noqa: PLR2004
                print_statement += (
                    "Long column names detected (>25 characters). Consider renaming "
                    f"the following columns {long_col_names}.",
                )

            print(print_statement)

    return data
