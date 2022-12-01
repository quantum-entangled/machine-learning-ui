from typing import Any, Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error
from Enums.SuccessMessages import Success


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    @property
    def columns(self) -> list:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def check_layer_capacity(
        self, layer_type: str, layer_name: str, num_columns: int
    ) -> bool:
        ...

    def add_columns(self, layer_type: str, layer_name: str, columns: Any) -> None:
        ...

    @property
    def input_layers(self) -> dict[str, Any]:
        ...

    @property
    def output_layers(self) -> dict[str, Any]:
        ...

    @property
    def input_shapes(self) -> dict[str, int]:
        ...

    @property
    def output_shapes(self) -> dict[str, int]:
        ...

    @property
    def layers_fullness(self) -> dict[str, int]:
        ...


class SelectModelColumnsWidget(iw.VBox):

    name = "Select Model Columns"

    def __init__(
        self, data_manager: DataManager, model_manager: ModelManager, **kwargs
    ) -> None:
        # Managers
        self.data_manager = data_manager
        self.model_manager = model_manager

        # Widgets
        self.layer_type_dropdown = iw.Dropdown(
            description="Choose layer type:", style={"description_width": "initial"}
        )
        self.layer_dropdown = iw.Dropdown(
            description="Choose layer:", style={"description_width": "initial"}
        )
        self.layer_fullness_status = iw.Label(value="Layer fullness: None")
        self.columns_select = iw.SelectMultiple(
            rows=10,
            description="Select columns:",
            style={"description_width": "initial"},
        )
        self.selected_columns_num = iw.Label(value="Selected: 0 column(s)")
        self.add_columns_button = iw.Button(description="Add Column(s)")
        self.columns_status = iw.Output()

        # Callbacks
        self.layer_type_dropdown.observe(
            self._on_layer_type_dropdown_value_change, names="value"
        )
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.columns_select.observe(self._on_columns_select_value_change, names="value")
        self.add_columns_button.on_click(self._on_add_columns_button_clicked)

        super().__init__(
            children=[
                self.layer_type_dropdown,
                self.layer_dropdown,
                self.layer_fullness_status,
                self.columns_select,
                self.selected_columns_num,
                self.add_columns_button,
                self.columns_status,
            ]
        )

        # Current State
        self.layer_type = self.layer_type_dropdown.value
        self.layer = self.layer_dropdown.value
        self.selected_columns = self.columns_select.value

    def _set_columns_select_options(self) -> None:
        self.columns_select.options = [
            item
            for item in self.columns_select.options
            if item not in self.selected_columns
        ]

    def _set_selected_columns_num(self) -> None:
        self.selected_columns_num.value = (
            f"Selected: {len(self.selected_columns)} column(s)"
        )

    def _set_layer_fullness_status(self) -> None:
        if self.layer_type == "input":
            shape = self.model_manager.input_shapes[self.layer]
        else:
            shape = self.model_manager.output_shapes[self.layer]

        current_num_columns = self.model_manager.layers_fullness[self.layer]

        self.layer_fullness_status.value = (
            f"Layer fullness: {current_num_columns}/{shape}"
        )

    def _on_layer_type_dropdown_value_change(self, change: Any) -> None:
        self.layer_type = change["new"]

        if change["new"] == "input":
            options = list(self.model_manager.input_layers)

            self.layer_dropdown.options = options
            self.layer_dropdown.value = options[0] if options else None
        else:
            options = list(self.model_manager.output_layers)

            self.layer_dropdown.options = options
            self.layer_dropdown.value = options[0] if options else None

    def _on_layer_dropdown_value_change(self, change: Any) -> None:
        self.layer = change["new"]

        self._set_layer_fullness_status()

    def _on_columns_select_value_change(self, change: Any) -> None:
        self.selected_columns = change["new"]

        self._set_selected_columns_num()

    def _on_add_columns_button_clicked(self, _) -> None:
        self.columns_status.clear_output(wait=True)

        with self.columns_status:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not self.model_manager.model_exists():
                print(Error.NO_MODEL_UPLOADED)
                return

            if not self.selected_columns:
                print(Error.NO_COLUMNS_SELECTED)
                return

            if not self.layer:
                print(Error.NO_LAYERS)
                return

            if not self.model_manager.check_layer_capacity(
                layer_type=self.layer_type,
                layer_name=self.layer,
                num_columns=len(self.selected_columns),
            ):
                print(Error.LAYER_OVERFILLED)
                return

            self.model_manager.add_columns(
                layer_type=self.layer_type,
                layer_name=self.layer,
                columns=self.selected_columns,
            )

            self._set_columns_select_options()
            self._set_selected_columns_num()
            self._set_layer_fullness_status()

            print(Success.COLUMNS_ADDED)

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.columns_status.clear_output()

        self.columns_select.options = self.data_manager.columns

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.columns_status.clear_output()

        self.layer_type_dropdown.options = ["input", "output"]
        self.layer_type_dropdown.value = "input"
