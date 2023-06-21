"""Utilities and auxiliary functions.

:author: Andreas Kanz

"""
from __future__ import annotations

from typing import Literal
from typing import TypedDict

import numpy as np
import pandas as pd


def _validate_input_bool(value: bool, desc: str) -> None:
    if not isinstance(value, bool):
        msg = f"Input value for '{desc}' is {type(value)} but should be a boolean."
        raise TypeError(msg)


def _validate_input_int(value: int, desc: str) -> None:
    if not isinstance(value, int):
        msg = f"Input value for '{desc}' is {type(value)} but should be an integer."
        raise TypeError(msg)


def _validate_input_range(value: int, desc: str, lower: int, upper: int) -> None:
    if value < lower or value > upper:
        msg = f"'{desc}' = {value} but should be {lower} <= '{desc}' <= {upper}."
        raise ValueError(msg)


def _validate_input_smaller(value1: int, value2: int, desc: str) -> None:
    if value1 > value2:
        msg = f"The first input for '{desc}' should be smaller or equal to the second."
        raise ValueError(msg)


def _validate_input_sum_smaller(limit: float, desc: str, *args) -> None:  # noqa: ANN002
    if sum(args) > limit:
        msg = f"The sum of input values for '{desc}' should be less or equal to {limit}."
        raise ValueError(msg)


def _validate_input_sum_larger(limit: float, desc: str, *args) -> None:  # noqa: ANN002
    if sum(args) < limit:
        msg = f"The sum of input values for '{desc}' should be larger/equal to {limit}."
        raise ValueError(msg)


def _validate_input_num_data(value: pd.DataFrame, desc: str) -> None:
    if value.select_dtypes(include=["number"]).empty:
        msg = f"Input value for '{desc}' should contain at least one numerical column."
        raise TypeError(msg)
