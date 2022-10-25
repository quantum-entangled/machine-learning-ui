from abc import abstractmethod
from typing import Any, Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser


class Manager(Protocol):
    """Protocol for model managers."""

    @abstractmethod
    def upload_model(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class UploadModelWidget(iw.VBox):
    """Widget to upload a model."""

    file_chooser_label = iw.Label(value="Please, select your model:")
    file_chooser = FileChooser(
        path="../db/Models", sandbox_path="../db/Models", filter_pattern="*.h5"
    )
    upload_model_button = iw.Button(description="Upload Model")
    upload_model_status = iw.Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the upload model widget window."""
        self._manager = manager
        self.upload_model_button.on_click(self._on_upload_model_button_clicked)

        super().__init__(
            children=[
                iw.HBox([self.file_chooser_label, self.file_chooser]),
                self.upload_model_button,
                self.upload_model_status,
            ],
            **kwargs
        )

    def _on_upload_model_button_clicked(self, _):
        """Callback for upload model button."""
        self._manager.upload_model(
            file_chooser=self.file_chooser, output_handler=self.upload_model_status
        )
