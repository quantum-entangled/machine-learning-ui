from dataclasses import dataclass
from typing import Any

import ipywidgets as iw
from Managers import ModelManager

from .CustomWidgets.ModelConfiguration import CreateModelWidget, UploadModelWidget


@dataclass
class Model:
    """The model container."""

    name: str | None = None
    instance: Any = None
    layers: dict | None = None


class ModelConfigurationUI(iw.VBox):
    """UI widgets for data preperation."""

    model = Model()
    manager = ModelManager(model=model)

    widget_children = [
        CreateModelWidget(manager=manager),
        UploadModelWidget(manager=manager),
    ]
    widget_titles = ["Model: Create", "Model: Upload"]
    widget = iw.Accordion(children=widget_children)
    for i, title in enumerate(widget_titles):
        widget.set_title(i, title)

    def __init__(self, **kwargs) -> None:
        """Initialize the main widget."""
        super().__init__(children=[self.widget], **kwargs)