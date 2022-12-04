from typing import Any

import ipywidgets as iw


class ConcatenateLayerWidget(iw.VBox):

    name = "Concatenate Layer"

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
        self.concatenate = iw.SelectMultiple(
            options=list(self.model_manager.layers),
            description="Select layers (at least 2):",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.layer_name, self.concatenate])

    def _update_widget(self) -> None:
        """Callback for widget update."""
        self.concatenate.options = list(self.model_manager.layers)

    @property
    def params(self) -> dict[str, Any]:
        return {"name": self.layer_name.value}

    @property
    def connect(self) -> list | int:
        return (
            [layer_name for layer_name in self.concatenate.value]
            if len(self.concatenate.value) >= 2
            else 0
        )
