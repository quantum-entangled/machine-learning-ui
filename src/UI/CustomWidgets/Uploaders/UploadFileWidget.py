from typing import Any, Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class DataManager(Protocol):
    """Protocol for data managers."""

    def upload_file(self, file_path: Any) -> None:
        ...


class UploadFileWidget(iw.VBox):
    """Widget to upload a file."""

    name = "Upload File"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
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
            file_path = self.file_chooser.selected

            if not file_path:
                print(Error.NO_FILE_PATH)
                return

            self.data_manager.upload_file(file_path=file_path)

            print(Success.FILE_UPLOADED)

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.upload_status.clear_output()
