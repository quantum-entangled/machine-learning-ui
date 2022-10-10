from ipywidgets import Tab

from .DataPreperationWidgets import DataGridWidget, UploadFile


class DataPreperationUI(Tab):
    def __init__(self, **kwargs):
        self.file_uploader = UploadFile()
        self.data_grid = DataGridWidget(file_uploader=self.file_uploader)

        self.tab_children = [self.file_uploader, self.data_grid]

        super().__init__(children=self.tab_children, **kwargs)
