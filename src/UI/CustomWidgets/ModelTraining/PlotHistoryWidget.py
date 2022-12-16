from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def plot_history(self, y: Any, color: Any, same_figure: bool) -> None:
        ...

    @property
    def training_history(self) -> dict[str, Any]:
        ...


class PlotHistoryWidget(iw.VBox):

    name = "Plot Training History"

    def __init__(self, model_manager: ModelManager, **kwargs):
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.history_type_dropdown = iw.Dropdown(
            description="Select history type:",
            style={"description_width": "initial"},
        )
        self.color_picker = iw.ColorPicker(
            description="Pick a line color:",
            value="blue",
            style={"description_width": "initial"},
        )
        self.same_figure = iw.Checkbox(
            value=False,
            description="Plot on same figure",
            style={"description_width": "initial"},
        )
        self.plot_button = iw.Button(description="Plot History")
        self.plot_output = iw.Output()

        # Callbacks
        self.plot_button.on_click(self._on_plot_button_clicked)

        super().__init__(
            children=[
                self.history_type_dropdown,
                self.color_picker,
                self.same_figure,
                self.plot_button,
                self.plot_output,
            ]
        )

    def _on_plot_button_clicked(self, _) -> None:
        """Callback for plot button."""
        self.plot_output.clear_output(wait=True)

        with self.plot_output:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.model_manager.training_history:
                print(Error.MODEL_NOT_TRAINED)
                return

            history_type = self.history_type_dropdown.value
            line_color = self.color_picker.value
            same_figure = self.same_figure.value

            self.model_manager.plot_history(
                y=history_type, color=line_color, same_figure=same_figure
            )

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.plot_output.clear_output()

    def _on_model_trained(self) -> None:
        """Callback for model training."""
        self.plot_output.clear_output()

        options = list(self.model_manager.training_history)

        self.history_type_dropdown.options = options
        self.history_type_dropdown.value = options[0] if options else None
