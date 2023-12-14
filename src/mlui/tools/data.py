import csv

import pandas as pd

from mlui.classes.errors import ParseCSVError, ValidateDataError
from mlui.types.classes import DataFrame


def parse_csv(csv_str: str) -> str:
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(csv_str).delimiter
    has_header = sniffer.has_header(csv_str)

    if delimiter not in (",", ";"):
        raise ParseCSVError("The file's delimiter is not ',' or ';'!")

    if not has_header:
        raise ParseCSVError("The file doesn't contain a header!")

    return delimiter


def validate_df(df: DataFrame) -> None:
    if isinstance(df.index, pd.MultiIndex):
        raise ValidateDataError("The DataFrame with MultiIndex is not supported!")

    if len(df.columns) < 2:
        raise ValidateDataError("The DataFrame contains less than 2 columns!")

    if len(df) < 2:
        raise ParseCSVError("The DataFrame contains less than 2 rows!")


def contains_nans(df: DataFrame) -> bool:
    return True if df.isna().values.any() else False


def contains_nonnumeric_dtypes(df: DataFrame) -> bool:
    nonnumeric_columns = df.select_dtypes(exclude=["float", "int"]).columns

    return True if len(nonnumeric_columns) != 0 else False
