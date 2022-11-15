from typing import Any, Protocol

import ipywidgets as iw
from IPython.display import display

from Enums.Callbacks import callbacks


class Manager(Protocol):
    """Protocol for training managers."""

    @property
    def model(self) -> Any:
        ...

    def add_callback(self, instance: Any, **kwargs) -> None:
        ...


class SelectCallbackWidget(iw.VBox):

    name = "Select Model Callbacks"

    def __init__(self, manager: Manager, **kwargs):
        self._manager = manager

        self.callback_dropdown = iw.Dropdown(
            options=list(callbacks),
            description="Select callback:",
            style={"description_width": "initial"},
        )
        self.callback_dropdown.observe(
            self._on_callback_dropdown_value_change, names="value"
        )
        self.callback_widget = iw.Output()
        self.add_callback_button = iw.Button(description="Add Callback")
        self.add_callback_button.on_click(self._on_add_callback_button_clicked)
        self.callback_status = iw.Output()

        self._current_callback = callbacks[self.callback_dropdown.value]
        self._current_callback_widget = self._current_callback.widget(
            manager=self._manager
        )
        self.callback_widget.append_display_data(self._current_callback_widget)

        super().__init__(
            children=[
                self.callback_dropdown,
                self.callback_widget,
                self.add_callback_button,
                self.callback_status,
            ],
            **kwargs,
        )

    def _on_callback_dropdown_value_change(self, change: Any) -> None:
        self.callback_widget.clear_output(wait=True)

        with self.callback_widget:
            self._current_callback = callbacks[change["new"]]

            if self._current_callback.widget is not None:
                self._current_callback_widget = self._current_callback.widget()
                display(self._current_callback_widget)
            else:
                self._current_callback_widget = None
                display(iw.Output())

    def _on_add_callback_button_clicked(self, _) -> None:
        self.callback_status.clear_output(wait=True)

        if not self._manager.model.instance:
            with self.callback_status:
                print("Please, upload the model first!\u274C")
            return

        if self._current_callback_widget is not None:
            self._manager.add_callback(
                instance=self._current_callback.instance,
                **self._current_callback_widget.params,
            )
        else:
            self._manager.add_callback(
                instance=self._current_callback.instance,
            )

        with self.callback_status:
            print(f"Callback has been successfully added!\u2705")
