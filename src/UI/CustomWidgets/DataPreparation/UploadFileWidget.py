from abc import abstractmethod
from typing import Any, Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser


class Manager(Protocol):
    """Protocol for data managers."""

    @abstractmethod
    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class UploadFileWidget(iw.VBox):
    """Widget to upload a file."""

    file_chooser_label = iw.Label(value="Please, select your data file:")
    file_chooser = FileChooser(
        path="../db/Datasets", sandbox_path="../db/Datasets", filter_pattern="*.txt"
    )
    upload_button = iw.Button(description="Upload File")
    upload_status = iw.Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the upload file widget window."""
        self._manager = manager

        self.upload_button.on_click(self._on_upload_button_clicked)

        super().__init__(
            children=[
                iw.HBox([self.file_chooser_label, self.file_chooser]),
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
