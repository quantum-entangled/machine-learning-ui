from typing import Any, Dict, List

import ipywidgets as iw
from Enums.Activations import activations


class DenseLayerWidget(iw.VBox):

    name = "Dense Layer"

    layer_name = iw.Text(
        value="",
        description="Layer name:",
        placeholder="Enter Layer Name",
        style={"description_width": "initial"},
    )
    units_num = iw.BoundedIntText(
        value=1,
        min=1,
        max=10_000,
        step=1,
        description="Number of units:",
        style={"description_width": "initial"},
    )
    activation = iw.Dropdown(
        options=list(activations),
        description="Activation function:",
        style={"description_width": "initial"},
    )
    connect_dropdown = iw.Dropdown(
        description="Connect layer to:",
        style={"description_width": "initial"},
    )

    def __init__(self, manager: Any, **kwargs) -> None:
        self._manager = manager

        self.connect_dropdown.options = tuple(self._manager.model.layers)

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
