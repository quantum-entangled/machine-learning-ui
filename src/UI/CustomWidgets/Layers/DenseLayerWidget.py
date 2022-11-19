from typing import Any, Dict, List

import ipywidgets as iw
from Enums.Activations import activations


class DenseLayerWidget(iw.VBox):

    name = "Dense Layer"

    def __init__(self, manager: Any, **kwargs) -> None:
        self._manager = manager

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
            options=list(self._manager.model.layers),
            description="Connect layer to:",
            style={"description_width": "initial"},
        )

        super().__init__(
            children=[
                self.layer_name,
                self.units_num,
                self.activation,
                self.connect_dropdown,
            ],
            **kwargs,
        )

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "name": self.layer_name.value,
            "units": self.units_num.value,
            "activation": activations[self.activation.value],
        }

    @property
    def connect(self) -> str | List | None:
        return self.connect_dropdown.value
