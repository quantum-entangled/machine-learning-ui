from typing import Any

import ipywidgets as iw
from Enums.Activations import activations


class DenseLayerWidget(iw.VBox):

    name = "Dense Layer"

    def __init__(self, model_manager: Any, **kwargs) -> None:
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.layer_name = iw.Text(
            value="",
            description="Layer name:",
            placeholder="Enter Layer Name",
            style={"description_width": "initial"},
        )
        self.units_num = iw.BoundedIntText(
            value=1,
            min=1,
            max=10_000,
            step=1,
            description="Number of units:",
            style={"description_width": "initial"},
        )
        self.activation = iw.Dropdown(
            options=list(activations),
            description="Activation function:",
            style={"description_width": "initial"},
        )
        self.connect_dropdown = iw.Dropdown(
            options=list(self.model_manager.layers),
            description="Connect layer to:",
            style={"description_width": "initial"},
        )

        super().__init__(
            children=[
                self.layer_name,
                self.units_num,
                self.activation,
                self.connect_dropdown,
            ]
        )

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.connect_dropdown.options = list(self.model_manager.layers)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "name": self.layer_name.value,
            "units": self.units_num.value,
            "activation": activations[self.activation.value],
        }

    @property
    def connect(self) -> str | int:
        return self.connect_dropdown.value if self.connect_dropdown.value else 0
