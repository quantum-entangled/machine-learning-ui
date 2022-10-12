import pandas as pd
from ipyfilechooser import FileChooser
from IPython.display import clear_output
from ipywidgets import Button, Label, Output, VBox, HBox


class UploadFileWidget(VBox):
    def __init__(
        self, data_file: object = None, widget_layout: object = None, **kwargs
    ):

        self.header = Label(value="Please, select your data file:")
        self.file_chooser = FileChooser(
            path="db/Datasets", sandbox_path="db", filter_pattern="*.txt"
        )
        self.upload_button = Button(description="Upload File")
        self.upload_status = Output()

        self.widget_children = [
            HBox([self.header, self.file_chooser], layout=widget_layout.layout),
            self.upload_button,
            self.upload_status,
        ]

        self.upload_button.on_click(self._upload_file(data_file))

        super().__init__(children=self.widget_children, **kwargs)

    def _get_file_path(self):
        return self.file_chooser.selected

    def _upload_file(self, data_file: object = None, *args):
        try:
            file_path = self._get_file_path()
            data_file.file = pd.read_csv(file_path, sep="  ", engine="python")

            with self.upload_status:
                clear_output()
                print("You're file is successfully uploaded!\u2705")
        except ValueError:
            with self.upload_status:
                clear_output()
                print("You're file is not selected!\u274C")
