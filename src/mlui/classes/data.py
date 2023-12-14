import io

import altair as alt
import pandas as pd
from altair import Chart

import mlui.tools as tools
from mlui.classes.errors import ParseCSVError, PlotError, UploadError, ValidateDataError
from mlui.types.classes import Columns, DataFrame


class Data:
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self._dataframe: DataFrame = pd.DataFrame()
        self.update_state()

    def update_state(self) -> None:
        self._columns: Columns = list(self._dataframe.columns)
        self._unused_columns: Columns = self._columns.copy()

    def upload(self, buff: io.BytesIO) -> None:
        """Read a file to the pandas dataframe.

        Parameters
        ----------
        buff : File-like object
            Buffer object to upload.

        Raises
        ------
        UploadError
            - If there are errors encountered during the parsing of the file.
            - If there are errors encountered during the execution of `pd.read_csv`
            function.
            - If there are errors encountered during the validation of the DataFrame.
        """
        try:
            csv_str = buff.getvalue().decode("utf-8")
            delimiter = tools.data.parse_csv(csv_str)
        except ParseCSVError as error:
            raise UploadError(error)

        try:
            df = pd.read_csv(buff, sep=delimiter, header=0, skipinitialspace=True)
        except (ValueError, pd.errors.ParserError) as error:
            raise UploadError(error)

        try:
            tools.data.validate_df(df)
        except ValidateDataError as error:
            raise UploadError(error)

        self._dataframe = df
        self.update_state()

    def set_unused_columns(self, available: list[str], selected: list[str]) -> None:
        used = dict.fromkeys(selected, True)
        unused_columns = [column for column in available if used.get(column) is None]

        self._unused_columns = unused_columns

    def get_unused_columns(self) -> Columns:
        return self._unused_columns.copy()

    def get_stats(self) -> DataFrame:
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

    def plot_columns(self, x: str | None, y: str | None, points: bool) -> Chart:
        if not x or not y:
            raise PlotError("Please, select the columns!")

        if x == y:
            columns = self._dataframe.loc[:, [x]].rename(columns={x: "Column"})
        else:
            columns = (
                self._dataframe.loc[:, [x, y]]
                .sort_values(by=x)
                .rename(columns={x: "Column_1", y: "Column_2"})
            )

        if tools.data.contains_nonnumeric_dtypes(columns):
            raise PlotError("Unable to plot columns of non-numeric dtype!")

        try:
            if x == y:
                chart = (
                    Chart(columns)
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
                    Chart(columns)
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
            raise PlotError("Unable to display the plot!")

        return chart

    @property
    def dataframe(self) -> DataFrame:
        return self._dataframe.copy()

    @property
    def columns(self) -> Columns:
        return self._columns.copy()

    @property
    def has_nans(self) -> bool:
        return tools.data.contains_nans(self._dataframe)

    @property
    def has_nonnumeric_dtypes(self) -> bool:
        return tools.data.contains_nonnumeric_dtypes(self._dataframe)

    @property
    def empty(self) -> bool:
        """Check if dataframe is empty."""
        return True if self._dataframe.empty else False
