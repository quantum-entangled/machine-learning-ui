from abc import abstractmethod
from typing import Any, Protocol

from ipyfilechooser import FileChooser
from ipywidgets import Button, HBox, Label, Output, VBox


class Manager(Protocol):
    """Protocol for data managers."""

    @abstractmethod
    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class UploadFileWidget(VBox):
    """Widget to upload a file."""

    file_chooser_label = Label(value="Please, select your data file:")
    file_chooser = FileChooser(
        path="../db/Datasets", sandbox_path="../db/Datasets", filter_pattern="*.txt"
    )
    upload_button = Button(description="Upload File")
    upload_status = Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the upload file widget window."""
        self._manager = manager

        self.upload_button.on_click(self._on_upload_button_clicked)

        super().__init__(
            children=[
                HBox([self.file_chooser_label, self.file_chooser]),
                self.upload_button,
                self.upload_status,
            ],
            **kwargs
        )

    def _on_upload_button_clicked(self, _):
        """Callback for upload file button."""
        self._manager.upload_file(
            file_chooser=self.file_chooser, output_handler=self.upload_status
        )
