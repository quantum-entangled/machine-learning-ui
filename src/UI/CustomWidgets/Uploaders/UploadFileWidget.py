from typing import Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser

from src.Enums.CautionMessages import Caution
from src.Enums.ErrorMessages import Error
from src.Enums.SuccessMessages import Success


class DataManager(Protocol):
    """Protocol for data managers."""

    def upload_file(self, file_path: str) -> None:
        ...

    def check_missing_values(self) -> list[str]:
        ...

    def check_non_numeric_columns(self) -> list[str]:
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
            path="db/Datasets",
            sandbox_path="db/Datasets",
            filter_pattern=["*.csv", "*.tsv"],
        )
        self.upload_file_button = iw.Button(description="Upload File")
        self.upload_status = iw.Output()

        # Callbacks
        self.upload_file_button.on_click(self._on_upload_file_button_clicked)

        super().__init__(
            children=[self.file_chooser, self.upload_file_button, self.upload_status]
        )

    def _on_upload_file_button_clicked(self, _) -> None:
        """Callback for upload file button."""
        self.upload_status.clear_output()

        with self.upload_status:
            file_path = self.file_chooser.selected

            if file_path is None:
                print(Error.NO_FILE_PATH)
                return

            self.data_manager.upload_file(file_path=file_path)

            missing_value_columns = self.data_manager.check_missing_values()

            if missing_value_columns:
                print(Caution.MISSING_VALUES, end=" ")
                print(*missing_value_columns, sep=",")

            non_numeric_columns = self.data_manager.check_non_numeric_columns()

            if non_numeric_columns:
                print(Caution.NON_NUMERIC, end=" ")
                print(*non_numeric_columns, sep=",")

            print(Success.FILE_UPLOADED)
