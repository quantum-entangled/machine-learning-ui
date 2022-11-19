from typing import Any, Dict, List

import ipywidgets as iw


class ConcatenateLayerWidget(iw.VBox):

    name = "Concatenate Layer"

    def __init__(self, manager: Any, **kwargs) -> None:
        self._manager = manager

        self.layer_name = iw.Text(
            value="",
            description="Layer name:",
            placeholder="Enter Layer Name",
            style={"description_width": "initial"},
        )
        self.concatenate = iw.SelectMultiple(
            options=list(self._manager.model.layers),
            description="Select layers (at least 2):",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.layer_name, self.concatenate], **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        return {"name": self.layer_name.value}

    @property
    def connect(self) -> str | List | None:
        return [layer_name for layer_name in self.concatenate.value]
