from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error
from src.Enums.Optimizers import optimizers
from src.Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def select_optimizer(self, optimizer: Any, **kwargs) -> None:
        ...


class SelectOptimizerWidget(iw.VBox):

    name = "Select Model Optimizer"

    def __init__(self, model_manager: ModelManager, **kwargs):
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.optimizer_dropdown = iw.Dropdown(
            options=list(optimizers),
            description="Select optimizer:",
            style={"description_width": "initial"},
        )
        self.optimizers_stack = iw.Stack(
            children=[optimizer.widget() for optimizer in optimizers.values()]
        )
        self.optimizer_widget = iw.Output()
        self.select_optimizer_button = iw.Button(description="Select Optimizer")
        self.optimizer_status = iw.Output()

        # Callbacks
        self.optimizer_dropdown.observe(
            self._on_optimizer_dropdown_value_change, names="value"
        )
        self.select_optimizer_button.on_click(self._on_select_optimizer_button_clicked)
        iw.jslink(
            (self.optimizer_dropdown, "index"),
            (self.optimizers_stack, "selected_index"),
        )

        super().__init__(
            children=[
                self.optimizer_dropdown,
                self.optimizers_stack,
                self.select_optimizer_button,
                self.optimizer_status,
            ]
        )

    def _on_optimizer_dropdown_value_change(self, _) -> None:
        self.optimizer_status.clear_output()

    def _on_select_optimizer_button_clicked(self, _) -> None:
        self.optimizer_status.clear_output(wait=True)

        with self.optimizer_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            optimizer = optimizers[self.optimizer_dropdown.value]
            optimizer_widget = self.optimizers_stack.children[
                self.optimizer_dropdown.index
            ]

            self.model_manager.select_optimizer(
                optimizer_=optimizer.instance, **optimizer_widget.params
            )

            print(Success.OPTIMIZER_SELECTED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.optimizer_status.clear_output()
