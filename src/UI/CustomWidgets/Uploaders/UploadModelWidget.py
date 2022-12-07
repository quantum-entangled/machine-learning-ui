from typing import Any, Protocol

import ipywidgets as iw
from ipyfilechooser import FileChooser

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    def upload_model(self, model_path: Any) -> None:
        ...


class UploadModelWidget(iw.VBox):
    """Widget to upload a model."""

    name = "Upload Model"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.file_chooser = FileChooser(
            title="Please, select the model:",
            path="../db/Models",
            sandbox_path="../db/Models",
            filter_pattern="*.h5",
        )
        self.upload_model_button = iw.Button(description="Upload Model")
        self.upload_status = iw.Output()

        # Callbacks
        self.upload_model_button.on_click(self._on_upload_model_button_clicked)

        super().__init__(
            children=[self.file_chooser, self.upload_model_button, self.upload_status]
        )

    def _on_upload_model_button_clicked(self, _) -> None:
        """Callback for upload model button."""
        self.upload_status.clear_output(wait=True)

        with self.upload_status:
            model_path = self.file_chooser.selected

            if not model_path:
                print(Error.NO_MODEL_PATH)
                return

            self.model_manager.upload_model(model_path=model_path)
            self.upload_status.clear_output()

            print(Success.MODEL_UPLOADED)
