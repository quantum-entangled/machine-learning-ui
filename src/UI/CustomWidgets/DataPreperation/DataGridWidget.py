from functools import partial
from typing import Any, Protocol

from ipydatagrid import DataGrid
from ipywidgets import Button, Output, VBox


class File(Protocol):
    """Protocol for data files."""

    file: Any


class DataGridWidget(VBox):
    """Widget to display a data grid."""

    show_grid_button = Button(description="Show Data Grid")
    grid_output = Output()

    widget_children = [show_grid_button, grid_output]

    def __init__(self, data_file: File, **kwargs) -> None:
        """Initialize the data grid widget window."""
        self.show_grid_button.on_click(partial(show_data_grid, data_file=data_file))

        super().__init__(children=self.widget_children, **kwargs)


@DataGridWidget.grid_output.capture(clear_output=True, wait=True)
def show_data_grid(*args, data_file: File) -> None:
    """Show data grid of the given file."""
    try:
        DataGrid(dataframe=data_file.file)
    except AttributeError:
        print("Please, upload the file first!\u274C")
