from typing import Any

import numpy as np
import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from ipyfilechooser import FileChooser
from IPython.display import display

from DataClasses import Data, Model
from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize the internal data object."""
        self._data = data
        self._model = model

    def upload_file(self, file_chooser: FileChooser) -> None:
        """Read file to the pandas format and store it."""

        file_path = file_chooser.selected

        if file_path is None:
            print(Error.NO_FILE_PATH)
            return

        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path, header=0, skipinitialspace=True
        )
        self._data.columns = list(self._data.file.columns)

        print(Success.FILE_UPLOADED)

    def show_data_grid(self) -> None:
        """Show the file data grid."""

        if self._data.file.empty:
            print(Error.NO_FILE_UPLOADED)
            return

        display(DataGrid(dataframe=self._data.file))

    def show_data_plot(self, x: Any, y: Any) -> None:
        """Show data plot."""

        if self._data.file.empty:
            print(Error.NO_FILE_UPLOADED)
            return

        x_data = self._data.file[x]
        y_data = self._data.file[y]

        fig = bqplt.figure()
        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1

        bqplt.plot(x=x_data, y=y_data, figure=fig)
        bqplt.xlabel(x)
        bqplt.ylabel(y)
        bqplt.show()

    def get_num_columns_per_layer(self, layer_name: str) -> int:
        if layer_name not in self._data.num_columns_per_layer.keys():
            self._data.num_columns_per_layer.update({layer_name: 0})

        return self._data.num_columns_per_layer[layer_name]

    def set_num_columns_per_layer(self, layer_name: str, num_columns: int) -> None:
        self._data.num_columns_per_layer[layer_name] += num_columns

    def add_model_columns(
        self, layer_type: str, layer_name: str, from_column: Any, to_column: Any
    ) -> None:
        if layer_type == "input":
            if layer_name not in self._data.input_training_columns.keys():
                self._data.input_training_columns[layer_name] = list()

            self._data.input_training_columns[layer_name] = sorted(
                set(
                    self._data.input_training_columns[layer_name]
                    + list(range(from_column, to_column))
                )
            )
        else:
            if layer_name not in self._data.output_training_columns.keys():
                self._data.output_training_columns[layer_name] = list()

            self._data.output_training_columns[layer_name] = sorted(
                set(
                    self._data.output_training_columns[layer_name]
                    + list(range(from_column, to_column))
                )
            )

    @property
    def data(self) -> Data:
        return self._data

    @property
    def columns(self) -> list:
        return self._data.columns
