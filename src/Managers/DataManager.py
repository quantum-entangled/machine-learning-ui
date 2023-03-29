import re
from csv import Sniffer
from typing import Any

import numpy as np
import pandas as pd
from bqplot import pyplot as bqplt
from ipydatagrid import DataGrid
from IPython.display import display
from sklearn.model_selection import train_test_split

from src.DataClasses import Data, Model
from src.Enums.ObserveTypes import Observe


class DataManager:
    """Manager for data preparation.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """

    def __init__(self, data: Data, model: Model) -> None:
        self._data = data
        self._model = model
        self._observers = list()

    def upload_file(self, file_path: str) -> None:
        """Read file to pandas format.

        Parameters
        ----------
        file_path : str
            Path to a file location.
        """
        with open(file_path, "r") as csvfile:
            value_delimiter = Sniffer().sniff(csvfile.readline()).delimiter
            decimal_delimiter = re.search(
                r"[^0-9\\" + value_delimiter + "]", csvfile.readline()
            )[0]

        if decimal_delimiter == "":
            decimal_delimiter = "."

        # Find first non-digit and non-delimiter as decimal separator, dot
        # by default
        self._data.file = pd.read_csv(
            filepath_or_buffer=file_path,
            header=0,
            skipinitialspace=True,
            sep=value_delimiter,
            decimal=decimal_delimiter,
        )
        self.refresh_data()
        self.notify_observers(callback_type=Observe.FILE)

    def check_missing_values(self) -> list[str]:
        """Check if file contains missing values.

        Returns
        -------
        list[str]
            List of names of columns with missing values.
        """
        return self._data.file.columns[self._data.file.isna().any()].to_list()

    def check_non_numeric_columns(self) -> list[str]:
        """Check if file contains non-numeric values.

        Returns
        -------
        list[str]
            List of names of columns with non-numeric values.
        """
        return self._data.file.select_dtypes(exclude="number").columns.to_list()

    def refresh_data(self) -> None:
        """Refresh attributes of data container."""
        self._data.columns = list(self._data.file.columns)
        self._data.input_columns = {name: list() for name in self._model.input_layers}
        self._data.output_columns = {name: list() for name in self._model.output_layers}
        self._data.columns_per_layer = {
            name: 0 for name in self._model.input_layers | self._model.output_layers
        }
        self._data.input_train_data = dict()
        self._data.output_train_data = dict()
        self._data.input_test_data = dict()
        self._data.output_test_data = dict()

    def show_data_grid(
        self, begin_row: int, end_row: int, begin_col: int, end_col: int
    ) -> None:
        """Show data grid.

        Parameters
        ----------
        begin_row : int
            Display from this row.
        end_row : int
            Display to this row.
        begin_col : int
            Display from this column.
        end_col : int
            Display to this column.
        """
        display(
            DataGrid(
                dataframe=self._data.file.iloc[begin_row:end_row, begin_col:end_col]
            )
        )

    def show_data_stats(self) -> None:
        """Show data statistics."""
        data_stat = pd.concat(
            [
                self._data.file.describe().transpose(),
                self._data.file.dtypes.rename("type"),
                pd.Series(
                    self._data.file.isnull().mean().round(3).mul(100), name="% of nulls"
                ),
            ],
            axis=1,
        )

        display(DataGrid(dataframe=data_stat))

    def show_data_plot(self, x: str, y: str) -> None:
        """Show data plot.

        Parameters
        ----------
        x : str
            X-axis column name.
        y : str
            Y-axis column name.
        """
        x_data = self._data.file[x]
        y_data = self._data.file[y]

        fig = bqplt.figure()
        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1
        fig.fig_margin = {"top": 5, "bottom": 35, "left": 45, "right": 5}

        if x == y:
            bqplt.hist(sample=x_data, figure=fig)
            bqplt.title(x)
        else:
            bqplt.plot(x=x_data, y=y_data, figure=fig, marker="circle", markersize=3)
            bqplt.xlabel(x)
            bqplt.ylabel(y)
        bqplt.show()

    def add_columns(
        self, layer_type: str, layer: str, columns: str | list[str]
    ) -> None:
        """Specify columns for model layers.

        Parameters
        ----------
        layer_type : str
            Type of layer for columns to be added.
        layer : str
            Name of layer for columns to be added.
        columns : str | list[str]
            One or more columns to add.
        """
        if layer_type == "input":
            self._data.input_columns[layer].extend(columns)
            self.notify_observers(callback_type=Observe.INPUT_COLUMNS_ADDED)
        else:
            self._data.output_columns[layer].extend(columns)
            self.notify_observers(callback_type=Observe.OUTPUT_COLUMNS_ADDED)

        self._data.columns_per_layer[layer] += len(columns)

    def check_inputs_fullness(self) -> bool:
        """Check data fullness of input layers.

        Returns
        -------
        bool
            True if input layer is filled with data, False otherwise.
        """
        for layer in self._model.input_layers:
            if self._data.columns_per_layer[layer] < self._model.input_shapes[layer]:
                return False
        return True

    def check_outputs_fullness(self) -> bool:
        """Check data fullness of output layers.

        Returns
        -------
        bool
            True if output layer is filled with data, False otherwise.
        """
        for layer in self._model.output_layers:
            if self._data.columns_per_layer[layer] < self._model.output_shapes[layer]:
                return False
        return True

    def split_data(self, test_size: int) -> None:
        """Split data into train and test sets.

        Parameters
        ----------
        test_size : int
            Percent of test portion to split.
        """
        train, test = train_test_split(self._data.file, test_size=test_size / 100)

        self._data.input_train_data = {
            name: train[values].to_numpy()
            for name, values in self._data.input_columns.items()
        }
        self._data.output_train_data = {
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

        self.notify_observers(callback_type=Observe.DATA_SPLIT)

    def notify_observers(self, callback_type: str) -> None:
        """Notify manager's observers with a specified callback.

        Parameters
        ----------
        callback_type : str
            A callback to invoke inside an observer.
        """
        for observer in self._observers:
            callback = getattr(observer, callback_type, None)

            if callable(callback):
                callback()

    def file_exists(self) -> bool:
        """Check if file exists.

        Returns
        -------
        bool
            True if file exists, False otherwise.
        """
        return False if self._data.file.empty else True

    @property
    def file(self) -> pd.DataFrame:
        """Pandas DataFrame containing the data."""
        return self._data.file

    @property
    def columns(self) -> list[str]:
        """List of data columns."""
        return self._data.columns

    @property
    def input_columns(self) -> dict[str, list[str]]:
        """Dictionary of columns bound to each input layer."""
        return self._data.input_columns

    @property
    def output_columns(self) -> dict[str, list[str]]:
        """Dictionary of columns bound to each output layer."""
        return self._data.output_columns

    @property
    def columns_per_layer(self) -> dict[str, int]:
        """Dictionary of columns' counters for each layer."""
        return self._data.columns_per_layer

    @property
    def input_train_data(self) -> dict[str, np.ndarray]:
        """Dictionary of train data portions for each input layer."""
        return self._data.input_train_data

    @property
    def output_train_data(self) -> dict[str, np.ndarray]:
        """Dictionary of train data portions for each output layer."""
        return self._data.output_train_data

    @property
    def input_test_data(self) -> dict[str, np.ndarray]:
        """Dictionary of test data portions for each input layer."""
        return self._data.input_test_data

    @property
    def output_test_data(self) -> dict[str, np.ndarray]:
        """Dictionary of test data portions for each output layer."""
        return self._data.output_test_data

    @property
    def observers(self) -> list[Any]:
        """List of manager's observers."""
        return self._observers

    @observers.setter
    def observers(self, observers_list: list[Any]) -> None:
        self._observers = observers_list
