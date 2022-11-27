from typing import Protocol

import ipywidgets as iw


class DataManager(Protocol):
    """Protocol for data managers."""

    def show_data_grid(self) -> None:
        ...


class DataGridWidget(iw.VBox):
    """Widget to display a data grid."""

    name = "Show Data Grid"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        self.data_manager = data_manager

        # Widgets
        self.show_grid_button = iw.Button(description="Show Data Grid")
        self.grid_output = iw.Output()

        # Callbacks
        self.show_grid_button.on_click(self._on_show_grid_button_clicked)

        super().__init__(children=[self.show_grid_button, self.grid_output])

    def _on_show_grid_button_clicked(self, _) -> None:
        """Callback for show data grid button."""
        self.grid_output.clear_output()

        with self.grid_output:
            self.data_manager.show_data_grid()

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.grid_output.clear_output()
