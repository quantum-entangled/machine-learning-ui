from typing import Any, Protocol

import ipywidgets as iw


class DataManager(Protocol):
    """Protocol for data managers."""

    def show_data_plot(self, x: Any, y: Any) -> None:
        ...

    @property
    def columns(self) -> list:
        ...


class DataPlotWidget(iw.VBox):
    """Widget to display a data plot."""

    name = "Show Data Plot"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data plot widget window."""
        self.data_manager = data_manager

        # Widgets
        self.x_dropdown = iw.Dropdown(description="Select X-axis:")
        self.y_dropdown = iw.Dropdown(description="Select Y-axis:")
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
        self.plot_output.clear_output()

        x = self.x_dropdown.value
        y = self.y_dropdown.value

        with self.plot_output:
            self.data_manager.show_data_plot(x=x, y=y)

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.plot_output.clear_output()

        if self.x_dropdown.options:
            return

        x_y_options = self.data_manager.columns

        if not x_y_options:
            return

        self.x_dropdown.options = self.y_dropdown.options = x_y_options
        self.x_dropdown.value = self.y_dropdown.value = x_y_options[0]
