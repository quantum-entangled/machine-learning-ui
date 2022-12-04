from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.Losses import losses
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def add_loss(self, layer: str, loss: Any) -> None:
        ...

    @property
    def losses(self) -> dict[str, Any]:
        ...


class SelectLossesWidget(iw.VBox):

    name = "Select Model Losses"

    def __init__(self, model_manager: ModelManager, **kwargs):
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.layer_dropdown = iw.Dropdown(
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.loss_dropdown = iw.Dropdown(
            options=list(losses),
            description="Select loss function:",
            style={"description_width": "initial"},
        )
        self.add_loss_button = iw.Button(description="Add Loss Function")
        self.loss_status = iw.Output()

        # Callbacks
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.add_loss_button.on_click(self._on_add_loss_button_clicked)

        super().__init__(
            children=[
                self.layer_dropdown,
                self.loss_dropdown,
                self.add_loss_button,
                self.loss_status,
            ]
        )

    def _on_layer_dropdown_value_change(self, _) -> None:
        self.loss_status.clear_output()

    def _on_add_loss_button_clicked(self, _) -> None:
        self.loss_status.clear_output(wait=True)

        with self.loss_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            layer = self.layer_dropdown.value
            loss = losses[self.loss_dropdown.value]

            if not layer:
                print(Error.NO_OUTPUT_LAYERS)
                return

            self.model_manager.add_loss(layer=layer, loss=loss)

            print(Success.LOSS_ADDED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.loss_status.clear_output()

        options = list(self.model_manager.losses)

        self.layer_dropdown.options = options
        self.layer_dropdown.value = options[0] if options else None
