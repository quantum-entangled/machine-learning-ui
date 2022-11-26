from typing import Any

import ipywidgets as iw
import numpy as np
import pandas as pd
from bqplot import Toolbar
from bqplot import pyplot as plt
from IPython.display import display

from DataClasses import Data, Model


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize the internal data object."""
        self._data = data
        self._model = model

    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        """Read file to the pandas format and store it."""
        output_handler.clear_output(wait=True)

        file_path = self.get_file_path(file_chooser=file_chooser)

        if file_path is None:
            with output_handler:
                print("Please, select the file first!\u274C")
            return

        self._data.file = np.loadtxt(file_path, skiprows=1)
        self._data.headers = list(
            pd.read_csv(file_path, nrows=1, header=0, sep="[ ]{1,}", engine="python")
        )

        with output_handler:
            print("Your file is successfully uploaded!\u2705")

    def get_file_path(self, file_chooser: Any) -> Any:
        """Get a data file path via the given file chooser."""
        return file_chooser.selected

    def show_data_grid(self, grid_class: Any, output_handler: Any) -> None:
        """Show the file data grid."""
        output_handler.clear_output(wait=True)

        with output_handler:
            if self._data.file is None:
                print("Please, upload the file first!\u274C")
                return

            df = pd.DataFrame(data=self._data.file, columns=self._data.headers)
            dg = grid_class(dataframe=df)
            display(dg)

    def show_data_plot(self, output_handler: Any) -> None:
        """Show data plot."""
        output_handler.clear_output(wait=True)

        if self._data.file is None:
            with output_handler:
                print("Please, upload the file first!\u274C")
            return

        headers = self._data.headers
        dropdown_options = [(header, pos) for pos, header in enumerate(headers)]
        x_dropdown = iw.Dropdown(description="x", options=dropdown_options, value=0)
        y_dropdown = iw.Dropdown(description="y", options=dropdown_options, value=0)

        fig = plt.figure()
        plt.plot(
            self._data.file[:, x_dropdown.value],
            self._data.file[:, y_dropdown.value],
            figure=fig,
        )
        plt.xlabel(headers[x_dropdown.value])
        plt.ylabel(headers[y_dropdown.value])

        def on_dropdown_value_change(*args):
            plt.current_figure().marks[0].x = self._data.file[:, x_dropdown.value]
            plt.current_figure().marks[0].y = self._data.file[:, y_dropdown.value]
            plt.xlabel(headers[x_dropdown.value])
            plt.ylabel(headers[y_dropdown.value])

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
