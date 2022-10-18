from dataclasses import dataclass
from typing import Any

from ipywidgets import Accordion

from .CustomWidgets.DataPreperation import UploadFileWidget


@dataclass
class DataFile:
    """The data file container."""

    file: Any = None


class DataPreperationUI(Accordion):
    """UI widgets for data preperation."""

    data_file = DataFile()

    widget_children = [
        UploadFileWidget(data_file),
    ]
    widget_titles = ("Upload File",)

    def __init__(self, **kwargs) -> None:
        """Initialize the main widget."""
        super().__init__(
            children=self.widget_children, titles=self.widget_titles, **kwargs
        )
