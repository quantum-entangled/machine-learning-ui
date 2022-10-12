from dataclasses import dataclass
from ipywidgets import VBox, Layout

from .DataPreperationWidgets import DataGridWidget, UploadFileWidget


@dataclass
class DataFile:
    file: object = None


@dataclass
class WidgetLayout:
    layout: object = Layout(
        height="auto", width="auto", justify_items="center", align_items="center"
    )


class DataPreperationUI(VBox):
    def __init__(self, **kwargs):
        data_file = DataFile()
        widget_layout = WidgetLayout()
        widget_children = [
            UploadFileWidget(data_file, widget_layout),
            # DataGridWidget(data_file, widget_layout),
        ]

        super().__init__(children=widget_children, **kwargs)
