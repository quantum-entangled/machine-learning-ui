from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def show_data_plot(self, x: Any, y: Any) -> None:
        ...

    @property
    def columns(self) -> list:
        ...


class DataPlotWidget(iw.VBox):
    """Widget to display a data plot."""

    name = "Show Data Plot"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.data_manager = data_manager

        # Widgets
        self.x_dropdown = iw.Dropdown(
            description="Select X-axis:", style={"description_width": "initial"}
        )
        self.y_dropdown = iw.Dropdown(
            description="Select Y-axis:", style={"description_width": "initial"}
        )
        self.show_plot_button = iw.Button(description="Show Data Plot")
        self.plot_output = iw.Output()

        # Callbacks
        self.show_plot_button.on_click(self._on_show_plot_button_clicked)

        super().__init__(
            children=[
                self.x_dropdown,
                self.y_dropdown,
                self.show_plot_button,
                self.plot_output,
            ]
        )

    def _on_show_plot_button_clicked(self, _) -> None:
        """Callback for show data plot button."""
        self.plot_output.clear_output(wait=True)

        with self.plot_output:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            x = self.x_dropdown.value
            y = self.y_dropdown.value

            self.data_manager.show_data_plot(x=x, y=y)

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.plot_output.clear_output()

        x_y_options = self.data_manager.columns

        self.x_dropdown.options = self.y_dropdown.options = x_y_options
        self.x_dropdown.value = self.y_dropdown.value = x_y_options[0]
