from typing import Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def save_model(self) -> None:
        ...


class SaveModelWidget(iw.VBox):
    """Widget to save a model."""

    name = "Save Model"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.save_model_button = iw.Button(description="Save Model")
        self.save_status = iw.Output()

        # Callbacks
        self.save_model_button.on_click(self._on_save_model_button_clicked)

        super().__init__(children=[self.save_model_button, self.save_status])

    def _on_save_model_button_clicked(self, _) -> None:
        """Callback for save model button."""
        self.save_status.clear_output(wait=True)

        with self.save_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            self.model_manager.save_model()
            self.save_status.clear_output()

            print(Success.MODEL_SAVED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.save_status.clear_output()
