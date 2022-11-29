from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    def create_model(self, model_name: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...


class CreateModelWidget(iw.VBox):
    """Widget to create a model."""

    name = "Create Model"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.model_name = iw.Text(
            value="",
            description="Model name:",
            placeholder="Enter Model Name",
        )
        self.create_model_button = iw.Button(description="Create Model")
        self.model_status = iw.Output()

        # Callbacks
        self.create_model_button.on_click(self._on_create_model_button_clicked)

        super().__init__(
            children=[
                self.model_name,
                self.create_model_button,
                self.model_status,
            ]
        )

    def _on_create_model_button_clicked(self, _) -> None:
        """Callback for create model button."""
        self.model_status.clear_output(wait=True)

        with self.model_status:
            model_name = self.model_name.value

            if not model_name:
                print(Error.NO_MODEL_NAME)
                return

            if self.model_manager.name == model_name:
                print(Error.SAME_MODEL_NAME)
                return

            self.model_manager.create_model(model_name=model_name)

            print(Success.MODEL_CREATED)

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.model_status.clear_output()
