import io

import altair as alt
import pandas as pd

import mlui.classes.errors as errors
import mlui.tools as tools
import mlui.types.classes as t


class Data:
    """
    Class representing a data file content.

    This class provides methods for managing and interacting with a DataFrame
    constructed from the data file.
    """

    def __init__(self) -> None:
        """Initialize an empty DataFrame."""
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset the state of the DataFrame.

        This method resets the internal state of the DataFrame to an empty object.
        """
        self._dataframe: t.DataFrame = pd.DataFrame()
        self.update_state()

    def update_state(self) -> None:
        """
        Update the internal state of the DataFrame, resetting its columns and unused
        columns.
        """
        self._columns: t.Columns = list(self._dataframe.columns)
        self._unused_columns: t.Columns = self._columns.copy()

    def upload(self, buff: io.BytesIO) -> None:
        """
        Upload data from a file into the DataFrame.

        Parameters
        ----------
        buff : file-like object
            Byte buffer containing the data.

        Raises
        ------
        UploadError
            If there is an issue parsing the file. If there is an issue reading the file
            to the DataFrame. If there is an issue validating the DataFrame.
        """
        try:
            csv_str = buff.getvalue().decode("utf-8")
            delimiter = tools.data.parse_csv(csv_str)
        except errors.ParseCSVError as error:
            raise errors.UploadError(error)

        try:
            df = pd.read_csv(buff, sep=delimiter, header=0, skipinitialspace=True)
        except (ValueError, pd.errors.ParserError) as error:
            raise errors.UploadError(error)

        try:
            tools.data.validate_df(df)
        except errors.ValidateDataError as error:
            raise errors.UploadError(error)

        self._dataframe = df
        self.update_state()

    def set_unused_columns(self, available: list[str], selected: list[str]) -> None:
        """
        Set the unused columns based on the available and selected columns.

        Parameters
        ----------
        available : list of str
            Available columns to choose from.
        selected : list of str
            Columns to set as used.

        Raises
        ------
        SetError
            If there is an issue setting the unused columns.
        """
        used = dict.fromkeys(selected, True)
        unused = [column for column in available if used.get(column) is None]

        self._unused_columns = unused

    def get_unused_columns(self) -> t.Columns:
        """
        Get the currently unused columns of the DataFrame.

        Returns
        -------
        list of str
            Currently unused columns.
        """
        return self._unused_columns.copy()

    def get_stats(self) -> t.DataFrame:
        """
        Get descriptive statistics and data types information for the DataFrame.

        Returns
        -------
        DataFrame
            DataFrame containing descriptive statistics and data types information.

        Raises
        ------
        PlotError
            If there is an issue generating the statistics.
        """
        try:
            stats = pd.concat(
                [
                    self._dataframe.describe().transpose(),
                    self._dataframe.dtypes.rename("dtype"),
                    pd.Series(
                        self._dataframe.isnull().mean().round(3).mul(100),
                        name="% of NULLs",
                    ),
                ],
                axis=1,
            )
        except ValueError:
            stats = pd.DataFrame()

        return stats

    def plot_columns(self, x: str | None, y: str | None, points: bool) -> t.Chart:
        """
        Plot columns from the DataFrame.

        Parameters
        ----------
        x : str or None
            Column to use for the x-axis.
        y : str or None
            Column to use for the y-axis.
        points : bool
            Whether to include points on the plot.

        Returns
        -------
        Chart
            Altair chart representing the plot.

        Raises
        ------
        PlotError
            If there is an issue generating the plot.
        """
        if not x or not y:
            raise errors.PlotError("Please, select the columns!")

        if x == y:
            columns = self._dataframe.loc[:, [x]].rename(columns={x: "Column"})
        else:
            columns = (
                self._dataframe.loc[:, [x, y]]
                .sort_values(by=x)
                .rename(columns={x: "Column_1", y: "Column_2"})
            )

        if tools.data.contains_nonnumeric_dtypes(columns):
            raise errors.PlotError("Unable to plot columns of non-numeric dtype!")

        try:
            if x == y:
                chart = (
                    alt.Chart(columns)
                    .mark_bar()
                    .encode(
                        x=alt.X("Column").title(x),
                        y=alt.Y("count()"),
                    )
                    .interactive(bind_x=True)
                    .properties(height=500)
                )
            else:
                chart = (
                    alt.Chart(columns)
                    .mark_line(point=points)
                    .encode(
                        x=alt.X("Column_1").scale(zero=False).title(x),
                        y=alt.Y("Column_2").scale(zero=False).title(y),
                        color=alt.Color().scale(scheme="set1"),
                    )
                    .interactive(bind_x=True, bind_y=True)
                    .properties(height=500)
                )
        except (ValueError, AttributeError, TypeError):
            raise errors.PlotError("Unable to display the plot!")

        return chart

    @property
    def dataframe(self) -> t.DataFrame:
        """Copy of the DataFrame."""
        return self._dataframe.copy()

    @property
    def columns(self) -> t.Columns:
        """Names of the columns in the DataFrame."""
        return self._columns.copy()

    @property
    def has_nans(self) -> bool:
        """True if there are NaN values in the DataFrame, False otherwise."""
        return tools.data.contains_nans(self._dataframe)

    @property
    def has_nonnumeric_dtypes(self) -> bool:
        """
        True if the DataFrame contains columns with non-numeric data types, False
        otherwise.
        """
        return tools.data.contains_nonnumeric_dtypes(self._dataframe)

    @property
    def empty(self) -> bool:
        """True if the DataFrame is empty, False otherwise."""
        return True if self._dataframe.empty else False
