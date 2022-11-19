from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for model managers."""

    def plot_model(self, output_handler: Any) -> None:
        ...


class PlotModelWidget(iw.VBox):
    """Widget to plot a model graph."""

    name = "Plot Model"

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the plot model widget window."""
        self._manager = manager

        self.plot_model_button = iw.Button(description="Plot Model")
        self.plot_model_button.on_click(self._on_plot_model_button_clicked)
        self.plot_model_output = iw.Output()

        super().__init__(
            children=[self.plot_model_button, self.plot_model_output], **kwargs
        )

    def _on_plot_model_button_clicked(self, _) -> None:
        """Callback for plot model button."""
        self._manager.plot_model(output_handler=self.plot_model_output)
