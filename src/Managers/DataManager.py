import itertools as it
from typing import Any

import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from IPython.display import display
from sklearn.model_selection import train_test_split

from DataClasses import Data, Model
from Enums.ObserveTypes import Observe


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize data object."""
        self._data = data
        self._model = model
        self._observers = list()

    def upload_file(self, file_path: Any) -> None:
        """Read file to pandas format."""
        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path, header=0, skipinitialspace=True
        )
        self.refresh_data()
        self.notify_observers(callback_type=Observe.FILE)

    def refresh_data(self) -> None:
        self._data.columns = list(self._data.file.columns)
        self._data.input_columns = {name: list() for name in self._model.input_layers}
        self._data.output_columns = {name: list() for name in self._model.output_layers}
        self._data.columns_per_layer = {
            name: 0 for name in self._model.input_layers | self._model.output_layers
        }
        self._data.input_training_data = dict()
        self._data.output_training_data = dict()
        self._data.input_test_data = dict()
        self._data.output_test_data = dict()

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

    def add_columns(self, layer_type: str, layer: str, columns: Any) -> None:
        if layer_type == "input":
            self._data.input_columns[layer].extend(columns)
            self.notify_observers(callback_type=Observe.INPUT_COLUMNS_ADDED)
        else:
            self._data.output_columns[layer].extend(columns)
            self.notify_observers(callback_type=Observe.OUTPUT_COLUMNS_ADDED)

        self._data.columns_per_layer[layer] += len(columns)

        if self.check_layers_fullness():
            self.notify_observers(callback_type=Observe.LAYERS_FILLED)

    def check_layers_fullness(self) -> bool:
        for layer in list(self._model.input_layers | self._model.output_layers):
            if layer in self._model.input_shapes:
                shape = self._model.input_shapes[layer]
            else:
                shape = self._model.output_shapes[layer]

            if self._data.columns_per_layer[layer] < shape:
                return False

        return True

    def split_data(self, test_size: int) -> None:
        train, test = train_test_split(self._data.file, test_size=test_size / 100)

        self._data.input_training_data = {
            name: train[values].to_numpy()
            for name, values in self._data.input_columns.items()
        }
        self._data.output_training_data = {
            name: train[values].to_numpy()
            for name, values in self._data.output_columns.items()
        }
        self._data.input_test_data = {
            name: test[values].to_numpy()
            for name, values in self._data.input_columns.items()
        }
        self._data.output_test_data = {
            name: test[values].to_numpy()
            for name, values in self._data.output_columns.items()
        }

    def notify_observers(self, callback_type: Any) -> None:
        for observer in self._observers:
            callback = getattr(observer, callback_type, None)

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
    def input_columns(self) -> dict[str, list[str]]:
        return self._data.input_columns

    @property
    def output_columns(self) -> dict[str, list[str]]:
        return self._data.output_columns

    @property
    def columns_per_layer(self) -> dict[str, int]:
        return self._data.columns_per_layer

    @property
    def observers(self) -> list[Any]:
        return self._observers

    @observers.setter
    def observers(self, observers_list: list[Any]) -> None:
        self._observers = observers_list
