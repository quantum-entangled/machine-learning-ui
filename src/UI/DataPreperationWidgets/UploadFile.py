import pandas as pd
from ipyfilechooser import FileChooser
from IPython.display import clear_output
from ipywidgets import Button, Label, Layout, Output, VBox


class UploadFile(VBox):
    def __init__(self, **kwargs):

        self.header = Label(value="Please, select your data file:")
        self.file_chooser = FileChooser(
            path="db/Datasets", sandbox_path="db", filter_pattern="*.txt"
        )
        self.upload_button = Button(description="Upload File")
        self.upload_status = Output()
        self.file = None
        self.widget_children = [
            self.header,
            self.file_chooser,
            self.upload_button,
            self.upload_status,
        ]
        self.widget_layout = Layout(
            height="auto", width="auto", justify_items="center", align_items="center"
        )
        self.widget_label = "Upload File"

        self.upload_button.on_click(self.__upload_file)

        super().__init__(
            children=self.widget_children, layout=self.widget_layout, **kwargs
        )

    def __get_file_path(self):
        return self.file_chooser.selected

    def __upload_file(self, *args):
        try:
            file_path = self.__get_file_path()
            self.file = pd.read_csv(file_path, sep="  ", engine="python")
            with self.upload_status:
                clear_output()
                print("You're file is successfully uploaded!\u2705")
        except ValueError:
            with self.upload_status:
                clear_output()
                print("You're file is not selected!\u274C")
