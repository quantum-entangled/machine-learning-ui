from typing import Any, Protocol

import ipywidgets as iw
from ipydatagrid import DataGrid


class Manager(Protocol):
    """Protocol for data managers."""

    def show_data_grid(self, grid_class: Any, output_handler: Any) -> None:
        ...


class DataGridWidget(iw.VBox):
    """Widget to display a data grid."""

    name = "Show Data Grid"

    show_grid_button = iw.Button(description="Show Data Grid")
    grid_output = iw.Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        self._manager = manager

        self.show_grid_button.on_click(self._on_show_grid_button_clicked)

        super().__init__(children=[self.show_grid_button, self.grid_output], **kwargs)

    def _on_show_grid_button_clicked(self, _) -> None:
        self._manager.show_data_grid(
            grid_class=DataGrid, output_handler=self.grid_output
        )
