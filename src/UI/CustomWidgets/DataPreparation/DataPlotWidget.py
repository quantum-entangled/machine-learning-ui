from abc import abstractmethod
from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for data managers."""

    @abstractmethod
    def show_data_plot(self, output_handler: Any) -> None:
        ...


class DataPlotWidget(iw.VBox):
    """Widget to display a data plot."""

    name = "Show Data Plot"

    show_plot_button = iw.Button(description="Show Data Plot")
    plot_output = iw.Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the data plot widget window."""
        self._manager = manager

        self.show_plot_button.on_click(self._on_show_lot_button_clicked)

        super().__init__(children=[self.show_plot_button, self.plot_output], **kwargs)

    def _on_show_lot_button_clicked(self, _):
        self._manager.show_data_plot(output_handler=self.plot_output)
