from typing import Any, Protocol

import ipywidgets as iw

from Enums.Losses import losses


class ModelManager(Protocol):
    """Protocol for training managers."""

    @property
    def model(self) -> Any:
        ...

    def add_loss(self, layer_name: str, loss: Any) -> None:
        ...


class SelectLossesWidget(iw.VBox):

    name = "Select Model Losses"

    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager

        self.layer_dropdown = iw.Dropdown(
            options=list(self.model_manager.model.output_names),
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.loss_dropdown = iw.Dropdown(
            options=list(losses),
            description="Choose loss function:",
            style={"description_width": "initial"},
        )
        self.add_loss_button = iw.Button(description="Add Loss Function")
        self.add_loss_button.on_click(self._on_add_loss_button_clicked)
        self.loss_status = iw.Output()

        super().__init__(
            children=[
                self.layer_dropdown,
                self.loss_dropdown,
                self.add_loss_button,
                self.loss_status,
            ],
            **kwargs,
        )

    def _on_add_loss_button_clicked(self, _) -> None:
        self.loss_status.clear_output(wait=True)

        if not self.model_manager.model.instance:
            with self.loss_status:
                print("Please, upload the model first!\u274C")
            return

        if not self.layer_dropdown.value:
            with self.loss_status:
                print("There are no output layers in the model!\u274C")
            return

        self.model_manager.add_loss(
            layer_name=self.layer_dropdown.value,
            loss=losses[self.loss_dropdown.value](),
        )

        with self.loss_status:
            print(f"Loss function has been successfully added!\u2705")

    def _on_widget_state_change(self) -> None:
        self.loss_status.clear_output(wait=True)

        options = self.model_manager.model.output_names

        if options:
            self.layer_dropdown.options = options
            self.layer_dropdown.value = options[0]
