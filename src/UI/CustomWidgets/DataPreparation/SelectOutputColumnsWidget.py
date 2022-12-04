import itertools as it
from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class DataManager(Protocol):
    """Protocol for data managers."""

    def refresh_data(self) -> None:
        ...

    def file_exists(self) -> bool:
        ...

    def add_columns(self, layer_type: str, layer: str, columns: Any) -> None:
        ...

    @property
    def columns(self) -> list:
        ...

    @property
    def input_columns(self) -> dict[str, list[str]]:
        ...

    @property
    def columns_per_layer(self) -> dict[str, int]:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def check_layer_capacity(
        self, layer_type: str, layer: str, num_columns: int
    ) -> bool:
        ...

    @property
    def output_layers(self) -> dict[str, Any]:
        ...

    @property
    def output_shapes(self) -> dict[str, int]:
        ...


class SelectOutputColumnsWidget(iw.VBox):

    name = "Select Output Columns"

    def __init__(
        self, data_manager: DataManager, model_manager: ModelManager, **kwargs
    ) -> None:
        # Managers
        self.data_manager = data_manager
        self.model_manager = model_manager

        # Widgets
        self.layer_dropdown = iw.Dropdown(
            description="Select layer:", style={"description_width": "initial"}
        )
        self.layer_fullness_status = iw.Label(value="Layer fullness: None/None")
        self.columns_select = iw.SelectMultiple(
            rows=10,
            description="Select columns:",
            style={"description_width": "initial"},
        )
        self.selected_columns_num = iw.Label(value="Selected: 0 column(s)")
        self.add_columns_button = iw.Button(description="Add Column(s)")
        self.columns_status = iw.Output()

        # Callbacks
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.columns_select.observe(self._on_columns_select_value_change, names="value")
        self.add_columns_button.on_click(self._on_add_columns_button_clicked)

        super().__init__(
            children=[
                self.layer_dropdown,
                self.layer_fullness_status,
                self.columns_select,
                self.selected_columns_num,
                self.add_columns_button,
                self.columns_status,
            ]
        )

    def _on_layer_dropdown_value_change(self, change: Any) -> None:
        self.layer_fullness_status.value = f"Layer fullness: {self.data_manager.columns_per_layer.get(change['new'])}/{self.model_manager.output_shapes.get(change['new'])}"

    def _on_columns_select_value_change(self, change: Any) -> None:
        self.selected_columns_num.value = f"Selected: {len(change['new'])} column(s)"

    def _on_add_columns_button_clicked(self, _) -> None:
        self.columns_status.clear_output(wait=True)

        with self.columns_status:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not self.model_manager.model_exists():
                print(Error.NO_MODEL_UPLOADED)
                return

            layer = self.layer_dropdown.value
            selected_columns = self.columns_select.value

            if not selected_columns:
                print(Error.NO_COLUMNS_SELECTED)
                return

            if not layer:
                print(Error.NO_LAYERS)
                return

            if not self.model_manager.check_layer_capacity(
                layer_type="output",
                layer=layer,
                num_columns=len(selected_columns),
            ):
                print(Error.LAYER_OVERFILLED)
                return

            self.data_manager.add_columns(
                layer_type="output",
                layer=layer,
                columns=selected_columns,
            )

            self.columns_select.options = [
                item
                for item in self.columns_select.options
                if item not in selected_columns
            ]
            self.selected_columns_num.value = "Selected: 0 column(s)"
            self.layer_fullness_status.value = f"Layer fullness: {self.data_manager.columns_per_layer.get(layer)}/{self.model_manager.output_shapes.get(layer)}"

            print(Success.COLUMNS_ADDED)

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.columns_status.clear_output()

        self.columns_select.options = self.data_manager.columns

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.columns_status.clear_output()

        self.data_manager.refresh_data()
        self.columns_select.options = self.data_manager.columns

        layers = list(self.model_manager.output_layers)

        self.layer_dropdown.options = layers
        self.layer_dropdown.value = layers[0] if layers else None

    def _on_input_columns_added(self) -> None:
        """Callback for adding input columns."""
        self.columns_select.options = [
            item
            for item in self.columns_select.options
            if item
            not in it.chain.from_iterable(self.data_manager.input_columns.values())
        ]
