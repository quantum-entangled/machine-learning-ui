from functools import partial
from typing import Any, Protocol

import numpy as np
import pandas as pd
from ipyfilechooser import FileChooser
from ipywidgets import Button, HBox, Label, Output, VBox


class File(Protocol):
    """Protocol for data files."""

    file: Any
    headers: Any


class UploadFileWidget(VBox):
    """Widget to upload a file."""

    file_chooser_label = Label(value="Please, select your data file:")
    file_chooser = FileChooser(
        path="../db/Datasets", sandbox_path="../db/Datasets", filter_pattern="*.txt"
    )
    upload_button = Button(description="Upload File")
    upload_status = Output()

    widget_children = [
        HBox([file_chooser_label, file_chooser]),
        upload_button,
        upload_status,
    ]

    def __init__(self, data_file: File, **kwargs) -> None:
        """Initialize the upload file widget window."""
        self.upload_button.on_click(
            partial(upload_file, data_file=data_file, file_chooser=self.file_chooser)
        )

        super().__init__(children=self.widget_children, **kwargs)


def get_file_path(file_chooser: FileChooser) -> Any:
    """Get the file path from file chooser object."""
    return file_chooser.selected


@UploadFileWidget.upload_status.capture(clear_output=True, wait=True)
def upload_file(*args, data_file: File, file_chooser: FileChooser) -> None:
    """Read file to the pandas format and store it in the global object."""
    try:
        file_path = get_file_path(file_chooser)
        data_file.file = np.loadtxt(file_path, skiprows=1)
        data_file.headers = list(
            pd.read_csv(file_path, nrows=1, header=0, sep="[ ]{1,}", engine="python")
        )

        print("Your file is successfully uploaded!\u2705")
    except ValueError:
        print("Please, select the file first!\u274C")
