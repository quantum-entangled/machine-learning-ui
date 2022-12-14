from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def plot_model(self) -> None:
        ...

    @property
    def output_layers(self) -> dict[str, Any]:
        ...


class PlotModelWidget(iw.VBox):
    """Widget to plot a model graph."""

    name = "Plot Model"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.plot_model_button = iw.Button(description="Plot Model")
        self.plot_output = iw.Output()

        # Callbacks
        self.plot_model_button.on_click(self._on_plot_model_button_clicked)

        super().__init__(children=[self.plot_model_button, self.plot_output])

    def _on_plot_model_button_clicked(self, _) -> None:
        """Callback for plot model button."""
        self.plot_output.clear_output(wait=True)

        with self.plot_output:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.model_manager.output_layers:
                print(Error.NO_MODEL_OUTPUTS)
                return

            self.model_manager.plot_model()

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.plot_output.clear_output()

    def _on_outputs_set(self) -> None:
        """Callback for setting outputs."""
        self.plot_output.clear_output()
