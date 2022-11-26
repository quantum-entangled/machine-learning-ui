from typing import Any, Dict, List

import ipywidgets as iw


class InputLayerWidget(iw.VBox):

    name = "Input Layer"

    def __init__(self, **kwargs) -> None:
        self.layer_name = iw.Text(
            value="",
            description="Layer name:",
            placeholder="Enter Layer Name",
            style={"description_width": "initial"},
        )
        self.input_shape = iw.BoundedIntText(
            value=1,
            min=1,
            max=10_000,
            step=1,
            description="Number of input columns:",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.layer_name, self.input_shape], **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        return {"name": self.layer_name.value, "shape": (self.input_shape.value,)}

    @property
    def connect(self) -> str | List | None:
        return None
