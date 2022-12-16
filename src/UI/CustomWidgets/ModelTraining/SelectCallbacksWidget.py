from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.Callbacks import callbacks
from src.Enums.ErrorMessages import Error
from src.Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def add_callback(self, callback: Any, **kwargs) -> None:
        ...

    @property
    def callbacks(self) -> list[Any]:
        ...


class SelectCallbacksWidget(iw.VBox):

    name = "Select Callbacks"

    def __init__(self, model_manager: ModelManager, **kwargs):
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.callback_dropdown = iw.Dropdown(
            options=list(callbacks),
            description="Select callback:",
            style={"description_width": "initial"},
        )
        self.callbacks_stack = iw.Stack(
            children=[
                callback.widget() for callback in callbacks.values() if callback.widget
            ]
        )
        self.add_callback_button = iw.Button(description="Add Callback")
        self.callback_status = iw.Output()

        # Callbacks
        self.callback_dropdown.observe(
            self._on_callback_dropdown_value_change, names="value"
        )
        self.add_callback_button.on_click(self._on_add_callback_button_clicked)
        iw.jslink(
            (self.callback_dropdown, "index"), (self.callbacks_stack, "selected_index")
        )

        super().__init__(
            children=[
                self.callback_dropdown,
                self.callbacks_stack,
                self.add_callback_button,
                self.callback_status,
            ]
        )

    def _on_callback_dropdown_value_change(self, _) -> None:
        self.callback_status.clear_output()

    def _on_add_callback_button_clicked(self, _) -> None:
        """Callback for add callback button."""
        self.callback_status.clear_output(wait=True)

        with self.callback_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            callback = callbacks[self.callback_dropdown.value]
            callback_widget = self.callbacks_stack.children[
                self.callback_dropdown.index
            ]

            if callback.instance in [
                type(callback) for callback in self.model_manager.callbacks
            ]:
                print(Error.SAME_CALLBACK)
                return

            if callback_widget:
                self.model_manager.add_callback(
                    callback=callback.instance, **callback_widget.params
                )
            else:
                self.model_manager.add_callback(callback=callback.instance)

            print(Success.CALLBACK_ADDED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.callback_status.clear_output()
