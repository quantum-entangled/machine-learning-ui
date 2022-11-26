from typing import Any, Protocol

import ipywidgets as iw


class ModelManager(Protocol):
    """Protocol for model managers."""

    def create_model(self, model_name: Any, output_handler: Any) -> None:
        ...


class CreateModelWidget(iw.VBox):
    """Widget to create a model."""

    name = "Create Model"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize the create model widget window."""
        self.model_manager = model_manager

        self.model_name = iw.Text(
            value="",
            description="Model name:",
            placeholder="Enter Model Name",
        )
        self.create_model_button = iw.Button(description="Create Model")
        self.create_model_button.on_click(self._on_create_model_button_clicked)
        self.create_model_status = iw.Output()

        super().__init__(
            children=[
                self.model_name,
                self.create_model_button,
                self.create_model_status,
            ],
            **kwargs
        )

    def _on_create_model_button_clicked(self, _) -> None:
        """Callback for create model button."""
        self.model_manager.create_model(
            model_name=self.model_name.value, output_handler=self.create_model_status
        )
