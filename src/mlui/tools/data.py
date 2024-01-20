import csv

import pandas as pd

import mlui.classes.errors as errors
import mlui.types.classes as t


def parse_csv(csv_str: str) -> str:
    try:
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(csv_str).delimiter
        has_header = sniffer.has_header(csv_str)
    except csv.Error:
        raise errors.ParseCSVError("The file's delimiter cannot be found!")

    if delimiter not in (",", ";"):
        raise errors.ParseCSVError("The file's delimiter is not ',' or ';'!")

    if not has_header:
        raise errors.ParseCSVError("The file doesn't contain a header!")

    return delimiter


def validate_df(df: t.DataFrame) -> None:
    if isinstance(df.index, pd.MultiIndex):
        raise errors.ValidateDataError(
            "The DataFrame with MultiIndex is not supported!"
        )

    if len(df.columns) < 2:
        raise errors.ValidateDataError("The DataFrame contains less than 2 columns!")

    if len(df) < 2:
        raise errors.ValidateDataError("The DataFrame contains less than 2 rows!")


def contains_nans(df: t.DataFrame) -> bool:
    return True if df.isna().values.any() else False


def contains_nonnumeric_dtypes(df: t.DataFrame) -> bool:
    nonnumeric_columns = df.select_dtypes(exclude=["float", "int"]).columns

    return True if len(nonnumeric_columns) != 0 else False
