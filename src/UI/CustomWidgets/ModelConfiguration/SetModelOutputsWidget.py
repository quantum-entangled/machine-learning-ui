from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error
from src.Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def set_model_outputs(self, outputs_names: str | list[str]) -> None:
        ...

    @property
    def input_layers(self) -> dict[str, Any]:
        ...

    @property
    def layers(self) -> dict[str, Any]:
        ...


class SetModelOutputsWidget(iw.VBox):
    """Widget to add and pop model layers."""

    name = "Set Model Outputs"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.outputs_select = iw.SelectMultiple(
            rows=5,
            description="Select model outputs:",
            style={"description_width": "initial"},
        )
        self.set_outputs_button = iw.Button(description="Set Outputs")
        self.outputs_status = iw.Output()

        # Callbacks
        self.set_outputs_button.on_click(self._on_set_outputs_button_clicked)

        super().__init__(
            children=[self.outputs_select, self.set_outputs_button, self.outputs_status]
        )

    def _set_outputs_select_options(self) -> None:
        self.outputs_select.options = [
            layer
            for layer in self.model_manager.layers
            if layer not in self.model_manager.input_layers
        ]

    def _on_set_outputs_button_clicked(self, _) -> None:
        """Callback for set outputs button."""
        self.outputs_status.clear_output(wait=True)

        with self.outputs_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            outputs_names = self.outputs_select.value

            if not outputs_names:
                print(Error.NO_MODEL_OUTPUTS)
                return

            self.model_manager.set_model_outputs(outputs_names=outputs_names)

            print(Success.OUTPUTS_SET)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.outputs_status.clear_output()

        self._set_outputs_select_options()

    def _on_layer_added(self) -> None:
        """Callback for adding layers to model."""
        self.outputs_status.clear_output()

        self._set_outputs_select_options()
