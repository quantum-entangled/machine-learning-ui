import io

import pandas as pd
from IPython.display import display
from ipywidgets import Tab

from .DataPreperationWidgets import *


class DataPreperationUI:
    def __init__(self) -> None:
        self.uploader = UploadFile()
        self.tab = Tab()
        self.tab.children = [self.uploader]

    def display(self):
        return self.tab

    def get_file_content(self):
        uploaded_file = self.uploader.value[0]
        file_content = pd.read_csv(io.BytesIO(uploaded_file.content))

        return display(file_content)
