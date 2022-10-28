from abc import abstractmethod
from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for model managers."""

    @abstractmethod
    def create_model(self, model_name: Any, output_handler: Any) -> None:
        ...


class CreateModelWidget(iw.VBox):
    """Widget to create a model."""

    model_name = iw.Text(
        value="",
        description="Model name:",
        placeholder="Enter Model Name",
    )
    create_model_button = iw.Button(description="Create Model")
    create_model_status = iw.Output()

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the create model widget window."""
        self._manager = manager

        self.create_model_button.on_click(self._on_create_model_button_clicked)

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
        self._manager.create_model(
            model_name=self.model_name.value, output_handler=self.create_model_status
        )
