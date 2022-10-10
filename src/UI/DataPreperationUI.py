from ipywidgets import Tab

from .DataPreperationWidgets import *


class DataPreperationUI(Tab):
    def __init__(self, **kwargs):
        self.uploader = UploadFile()

        self.tab_children = [self.uploader]

        super().__init__(children=self.tab_children, **kwargs)
