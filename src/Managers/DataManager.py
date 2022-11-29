from typing import Any

import numpy as np
import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from IPython.display import display

from DataClasses import Data, Model


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize data object."""
        self._data = data
        self._model = model

    def upload_file(self, file_path: Any) -> None:
        """Read file to pandas format."""
        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path, header=0, skipinitialspace=True
        )
        self._data.columns = list(self._data.file.columns)

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
