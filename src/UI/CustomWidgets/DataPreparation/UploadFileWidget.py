from typing import Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser


class DataManager(Protocol):
    """Protocol for data managers."""

    def upload_file(self, file_chooser: FileChooser) -> None:
        ...


class UploadFileWidget(iw.VBox):
    """Widget to upload a file."""

    name = "Upload File"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the upload file widget window."""
        self.data_manager = data_manager

        # Widgets
        self.file_chooser = FileChooser(
            title="Please, select the data file:",
            path="../db/Datasets",
            sandbox_path="../db/Datasets",
            filter_pattern=["*.csv", "*.tsv"],
        )
        self.upload_button = iw.Button(description="Upload File")
        self.upload_status = iw.Output()

        # Callbacks
        self.upload_button.on_click(self._on_upload_button_clicked)

        super().__init__(
            children=[self.file_chooser, self.upload_button, self.upload_status]
        )

    def _on_upload_button_clicked(self, _) -> None:
        """Callback for upload file button."""
        self.upload_status.clear_output()

        with self.upload_status:
            self.data_manager.upload_file(file_chooser=self.file_chooser)

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.upload_status.clear_output()
