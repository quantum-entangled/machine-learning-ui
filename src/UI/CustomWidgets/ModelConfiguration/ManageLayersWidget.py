from typing import Any, Protocol

import ipywidgets as iw
from IPython.display import display

from Enums.Layers import layers


class Manager(Protocol):
    """Protocol for model managers."""

    def add_layer(
        self,
        layer_type: Any,
        instance: Any,
        connect_to: Any,
        output_handler: Any,
        **kwargs,
    ) -> None:
        ...

    def show_model_summary(self, output_handler: Any) -> None:
        ...


class ManageLayersWidget(iw.VBox):
    """Widget to add and pop model layers."""

    name = "Manage Layers"

    def __init__(self, manager: Manager, **kwargs) -> None:
        """Initialize the manage layers widget window."""
        self._manager = manager

        self.layer_type_dropdown = iw.Dropdown(
            options=list(layers),
            description="Choose layer type:",
            style={"description_width": "initial"},
        )
        self.layer_type_dropdown.observe(
            self._on_layer_type_dropdown_value_change, names="value"
        )
        self.layer_widget_output = iw.Output()
        self.add_layer_button = iw.Button(description="Add Layer")
        self.add_layer_button.on_click(self._on_add_layer_button_clicked)
        self.layer_status = iw.Output()
        self.model_summary_output = iw.Output()

        self._current_layer = layers[self.layer_type_dropdown.value]
        self._current_layer_widget = self._current_layer.widget(manager=self._manager)
        self.layer_widget_output.append_display_data(self._current_layer_widget)

        super().__init__(
            children=[
                self.layer_type_dropdown,
                self.layer_widget_output,
                iw.HBox(children=[self.add_layer_button, self.layer_status]),
                self.model_summary_output,
            ],
            **kwargs,
        )

    def _on_layer_type_dropdown_value_change(self, change: Any) -> None:
        self.layer_widget_output.clear_output(wait=True)

        with self.layer_widget_output:
            self._current_layer = layers[change["new"]]
            self._current_layer_widget = self._current_layer.widget(
                manager=self._manager
            )
            display(self._current_layer_widget)

    def _on_add_layer_button_clicked(self, _) -> None:
        layer_type = self.layer_type_dropdown.value

        self._manager.add_layer(
            layer_type=layer_type,
            instance=self._current_layer.instance,
            connect_to=self._current_layer_widget.connect,
            output_handler=self.layer_status,
            **self._current_layer_widget.params,
        )

        self._manager.show_model_summary(output_handler=self.model_summary_output)

        self.layer_widget_output.clear_output(wait=True)

        with self.layer_widget_output:
            self._current_layer_widget = self._current_layer.widget(
                manager=self._manager
            )
            display(self._current_layer_widget)
