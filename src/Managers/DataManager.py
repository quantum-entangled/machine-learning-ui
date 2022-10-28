from typing import Any, Protocol

import numpy as np
import pandas as pd
from IPython.display import display


class File(Protocol):
    """Protocol for data files."""

    file: Any
    headers: Any


class DataManager:
    """Manager for data preperation."""

    def __init__(self, data: File) -> None:
        """Initialize the internal model object."""
        self._data = data

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

    @property
    def data(self) -> File:
        return self._data
