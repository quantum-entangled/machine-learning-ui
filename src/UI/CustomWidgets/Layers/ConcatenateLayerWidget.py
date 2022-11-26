from typing import Any

import ipywidgets as iw


class ConcatenateLayerWidget(iw.VBox):

    name = "Concatenate Layer"

    def __init__(self, **kwargs) -> None:
        self.layer_name = iw.Text(
            value="",
            description="Layer name:",
            placeholder="Enter Layer Name",
            style={"description_width": "initial"},
        )
        self.concatenate = iw.SelectMultiple(
            options=list(kwargs["model_layers"]),
            description="Select layers (at least 2):",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.layer_name, self.concatenate], **kwargs)

    @property
    def params(self) -> dict[str, Any]:
        return {"name": self.layer_name.value}

    @property
    def connect(self) -> str | list | None:
        return [layer_name for layer_name in self.concatenate.value]
