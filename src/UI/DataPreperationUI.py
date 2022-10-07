import io

import pandas as pd
from ipywidgets import Tab

from .DataPreperationWidgets import *


class DataPreperationUI(Tab):
    def __init__(self):
        self.uploader = UploadFile()
        self.children = [self.uploader]

        super().__init__(children=self.children)

    def _get_file_content(self):
        uploaded_file = self.uploader.value[0]
        file_content = pd.read_csv(io.BytesIO(uploaded_file.content))

        return file_content
