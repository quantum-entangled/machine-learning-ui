from typing import Any

import ipywidgets as iw
import numpy as np
import pandas as pd
from bqplot import Toolbar
from bqplot import pyplot as plt
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

        if self._data.file is None:
            print(Error.NO_FILE_UPLOADED)
            return

        display(DataGrid(dataframe=self._data.file))

    def show_data_plot(self, output_handler: Any) -> None:
        """Show data plot."""
        output_handler.clear_output(wait=True)

        if self._data.file is None:
            with output_handler:
                print("Please, upload the file first!\u274C")
            return

        columns = self._data.columns
        dropdown_options = [(header, pos) for pos, header in enumerate(columns)]
        x_dropdown = iw.Dropdown(description="x", options=dropdown_options, value=0)
        y_dropdown = iw.Dropdown(description="y", options=dropdown_options, value=0)

        fig = plt.figure()
        plt.plot(
            self._data.file[:, x_dropdown.value],
            self._data.file[:, y_dropdown.value],
            figure=fig,
        )
        plt.xlabel(columns[x_dropdown.value])
        plt.ylabel(columns[y_dropdown.value])

        def on_dropdown_value_change(*args):
            plt.current_figure().marks[0].x = self._data.file[:, x_dropdown.value]
            plt.current_figure().marks[0].y = self._data.file[:, y_dropdown.value]
            plt.xlabel(columns[x_dropdown.value])
            plt.ylabel(columns[y_dropdown.value])

        x_dropdown.observe(on_dropdown_value_change, names="value")
        y_dropdown.observe(on_dropdown_value_change, names="value")

        plot_window = iw.TwoByTwoLayout(
            top_left=iw.VBox([x_dropdown, y_dropdown]),
            top_right=iw.VBox([fig, Toolbar(figure=fig)]),
            align_items="center",
            height="auto",
            width="auto",
        )

        with output_handler:
            display(plot_window)

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
