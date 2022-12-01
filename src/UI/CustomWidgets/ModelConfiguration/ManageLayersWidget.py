from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.Layers import layers
from Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def add_layer(self, layer_instance: Any, connect_to: Any, **kwargs) -> None:
        ...

    @property
    def layers(self) -> dict[str, Any]:
        ...


class ManageLayersWidget(iw.VBox):
    """Widget to add and pop model layers."""

    name = "Manage Layers"

    def __init__(self, model_manager: ModelManager, **kwargs) -> None:
        """Initialize widget window."""
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.layer_type_dropdown = iw.Dropdown(
            options=list(layers),
            description="Choose layer type:",
            style={"description_width": "initial"},
        )
        self.layers_stack = iw.Stack(
            children=[
                layer.widget(model_manager=self.model_manager)
                for layer in layers.values()
            ]
        )
        self.add_layer_button = iw.Button(description="Add Layer")
        self.layer_status = iw.Output()

        # Callbacks
        self.add_layer_button.on_click(self._on_add_layer_button_clicked)
        iw.jslink(
            (self.layer_type_dropdown, "index"), (self.layers_stack, "selected_index")
        )

        super().__init__(
            children=[
                self.layer_type_dropdown,
                self.layers_stack,
                self.add_layer_button,
                self.layer_status,
            ]
        )

    def _update_layer_widget(self) -> None:
        for layer_widget in self.layers_stack.children:
            if callable(getattr(layer_widget, "_on_widget_state_change", None)):
                layer_widget._on_widget_state_change()

    def _on_add_layer_button_clicked(self, _) -> None:
        """Callback for add layer button."""
        self.layer_status.clear_output(wait=True)

        with self.layer_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            layer = layers[self.layer_type_dropdown.value]
            layer_widget = self.layers_stack.children[self.layer_type_dropdown.index]
            layer_name = layer_widget.params["name"]
            connect_to = layer_widget.connect

            if not layer_name:
                print(Error.NO_LAYER_NAME)
                return

            if layer_name in self.model_manager.layers:
                print(Error.SAME_LAYER_NAME)
                return

            if connect_to == 0:
                print(Error.NO_CONNECT_TO)
                return

            self.model_manager.add_layer(
                layer_instance=layer.instance,
                connect_to=connect_to,
                **layer_widget.params,
            )
            self._update_layer_widget()

            print(Success.LAYER_ADDED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.layer_status.clear_output()

        self._update_layer_widget()
