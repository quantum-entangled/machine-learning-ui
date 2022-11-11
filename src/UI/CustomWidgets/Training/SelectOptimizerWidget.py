from typing import Any, Protocol

import ipywidgets as iw
from IPython.display import display

from Enums.Optimizers import optimizers


class Manager(Protocol):
    """Protocol for training managers."""

    @property
    def model(self) -> Any:
        ...

    def select_optimizer(self, instance: Any, **kwargs) -> None:
        ...


class SelectOptimizerWidget(iw.VBox):

    name = "Select Model Optimizer"

    def __init__(self, manager: Manager, **kwargs):
        self._manager = manager

        self.optimizer_dropdown = iw.Dropdown(
            options=list(optimizers),
            description="Select optimizer:",
            style={"description_width": "initial"},
        )
        self.optimizer_dropdown.observe(
            self._on_optimizer_dropdown_value_change, names="value"
        )
        self.optimizer_widget = iw.Output()
        self.select_optimizer_button = iw.Button(description="Select Optimizer")
        self.select_optimizer_button.on_click(self._on_select_optimizer_button_clicked)
        self.optimizer_status = iw.Output()

        self._current_optimizer = optimizers[self.optimizer_dropdown.value]
        self._current_optimizer_widget = self._current_optimizer.widget(
            manager=self._manager
        )
        self.optimizer_widget.append_display_data(self._current_optimizer_widget)

        super().__init__(
            children=[
                self.optimizer_dropdown,
                self.optimizer_widget,
                self.select_optimizer_button,
                self.optimizer_status,
            ],
            **kwargs,
        )

    def _on_optimizer_dropdown_value_change(self, change: Any) -> None:
        self.optimizer_widget.clear_output(wait=True)

        with self.optimizer_widget:
            self._current_optimizer = optimizers[change["new"]]
            self._current_optimizer_widget = self._current_optimizer.widget()
            display(self._current_optimizer_widget)

    def _on_select_optimizer_button_clicked(self, _) -> None:
        self.optimizer_status.clear_output(wait=True)

        if not self._manager.model.instance:
            with self.optimizer_status:
                print("Please, upload the model first!\u274C")
            return

        self._manager.select_optimizer(
            instance=self._current_optimizer.instance,
            **self._current_optimizer_widget.params,
        )

        with self.optimizer_status:
            print(f"Optimizer has been successfully selected!\u2705")
