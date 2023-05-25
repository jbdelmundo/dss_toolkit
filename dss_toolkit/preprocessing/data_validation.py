import numpy as np


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
