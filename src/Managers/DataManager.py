from typing import Any

import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from IPython.display import display

from DataClasses import Data, Model
from Enums.WatchTypes import Watch


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize data object."""
        self._data = data
        self._model = model
        self._watchers = list()

    def upload_file(self, file_path: Any) -> None:
        """Read file to pandas format."""
        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path, header=0, skipinitialspace=True
        )
        self._data.columns = list(self._data.file.columns)

        self.callback_watchers(callback_type=Watch.FILE)

    def show_data_grid(self) -> None:
        """Show data grid."""
        display(DataGrid(dataframe=self._data.file))

    def show_data_plot(self, x: Any, y: Any) -> None:
        """Show data plot."""
        x_data = self._data.file[x]
        y_data = self._data.file[y]

        fig = bqplt.figure()
        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1

        bqplt.plot(x=x_data, y=y_data, figure=fig)
        bqplt.xlabel(x)
        bqplt.ylabel(y)
        bqplt.show()

    def callback_watchers(self, callback_type: Any) -> None:
        for watcher in self._watchers:
            callback = getattr(watcher, callback_type, None)

            if callable(callback):
                callback()

    def file_exists(self) -> bool:
        return False if self._data.file.empty else True

    @property
    def data(self) -> Data:
        return self._data

    @property
    def file(self) -> pd.DataFrame:
        return self._data.file

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def watchers(self) -> list[Any]:
        return self._watchers

    @watchers.setter
    def watchers(self, watchers_list: list[Any]) -> None:
        self._watchers = watchers_list
