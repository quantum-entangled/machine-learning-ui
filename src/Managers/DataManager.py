from typing import Any,Protocol

import numpy as np
import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from IPython.display import display

from DataClasses import Data, Model


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize the internal data object."""
        self._data = data
        self._model = model
        """row_begin, row_end, columns begin, columns end"""
        self._range = [0, 0, 0, 0]


    def upload_file(self, file_path: Any) -> None:
        """Read file to the pandas format and store it."""
        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path, header=0, skipinitialspace=True
        )
        self._data.columns = list(self._data.file.columns)
    def set_range_grid(self, begin_r: Protocol, end_r: Protocol,
                       begin_c: Protocol, end_c: Protocol, output_handler: Any) -> None:
        self._range[0] = begin_r
        self._range[1] = end_r
        self._range[2] = begin_c
        self._range[3] = end_c
        with output_handler:
            out_txt="Rows from {0} to {1} and Columns from {2} to {3}"
            print(out_txt.format(self._range[0],
                                 self._range[1],
                                 self._range[2],
                                 self._range[3]))
    def show_data_grid(self) -> None:
        """Show the file data grid."""

        display(DataGrid(dataframe=self._data.file.iloc[self._range[0]:self._range[1],self._range[2]:self._range[3]]))

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
