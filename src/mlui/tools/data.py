import csv

import pandas as pd

import mlui.classes.errors as errors
import mlui.types.classes as t


def parse_csv(csv_str: str) -> str:
    """
    Parse the delimiter of a CSV string and check for a header.

    Parameters
    ----------
    csv_str : str
        CSV string to be parsed.

    Returns
    -------
    str
        Identified delimiter.

    Raises
    ------
    ParseCSVError
        If the delimiter or header cannot be determined. If the delimiter is not one of
        ',' or ';'.
    """
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
    """
    Validate the structure of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be validated.

    Raises
    ------
    ValidateDataError
        If the index of the DataFrame is an instance of `MultiIndex`. If the DataFrame
        contains less than 2 columns. If the DataFrame contains less than 2 rows.
    """
    if isinstance(df.index, pd.MultiIndex):
        raise errors.ValidateDataError(
            "The DataFrame with MultiIndex is not supported!"
        )

    if len(df.columns) < 2:
        raise errors.ValidateDataError("The DataFrame contains less than 2 columns!")

    if len(df) < 2:
        raise errors.ValidateDataError("The DataFrame contains less than 2 rows!")


def contains_nans(df: t.DataFrame) -> bool:
    """
    Check if a DataFrame contains any NaN values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be checked.

    Returns
    -------
    bool
        True if there are NaN values, False otherwise.
    """
    return True if df.isna().values.any() else False


def contains_nonnumeric_dtypes(df: t.DataFrame) -> bool:
    """
    Check if a DataFrame contains columns with non-numeric data types.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be checked.

    Returns
    -------
    bool
        True if there are non-numeric data types, False otherwise.
    """
    nonnumeric_columns = df.select_dtypes(exclude=["float", "int"]).columns

    return True if len(nonnumeric_columns) != 0 else False
