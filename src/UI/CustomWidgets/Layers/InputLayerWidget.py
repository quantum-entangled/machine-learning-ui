from typing import Any, Dict

import ipywidgets as iw


class InputLayerWidget(iw.VBox):
    layer_name = iw.Text(
        value="",
        description="Layer name:",
        placeholder="Enter Layer Name",
        style={"description_width": "initial"},
    )
    input_shape = iw.BoundedIntText(
        value=1,
        min=1,
        max=10_000,
        step=1,
        description="Number of input columns:",
        style={"description_width": "initial"},
    )

    def __init__(self, manager: Any, **kwargs) -> None:
        self._manager = manager

        super().__init__(children=[self.layer_name, self.input_shape], **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        return {"name": self.layer_name.value, "shape": (self.input_shape.value,)}
