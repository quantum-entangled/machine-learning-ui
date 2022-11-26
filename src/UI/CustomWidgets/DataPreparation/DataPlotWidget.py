from typing import Any, Protocol

import ipywidgets as iw


class DataManager(Protocol):
    """Protocol for data managers."""

    def show_data_plot(self, output_handler: Any) -> None:
        ...


class DataPlotWidget(iw.VBox):
    """Widget to display a data plot."""

    name = "Show Data Plot"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data plot widget window."""
        self.data_manager = data_manager

        self.show_plot_button = iw.Button(description="Show Data Plot")
        self.plot_output = iw.Output()
        self.show_plot_button.on_click(self._on_show_plot_button_clicked)

        super().__init__(children=[self.show_plot_button, self.plot_output], **kwargs)

    def _on_show_plot_button_clicked(self, _) -> None:
        self.data_manager.show_data_plot(output_handler=self.plot_output)
