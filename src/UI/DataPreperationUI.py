import io

import pandas as pd
from IPython.display import display
from ipywidgets import Tab

from .DataPreperationWidgets import *


class DataPreperationUI(Tab):
    def __init__(self, **kwargs):
        self.uploader = UploadFile()
        self.tab_children = [self.uploader]

        super().__init__(self.tab_children, **kwargs)

    def get_file_content(self):
        uploaded_file = self.uploader.value[0]
        file_content = pd.read_csv(io.BytesIO(uploaded_file.content))

        return display(file_content)
