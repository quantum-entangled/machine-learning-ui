from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for training managers."""

    @property
    def data(self) -> Any:
        ...

    @property
    def model(self) -> Any:
        ...

    @property
    def config(self) -> Any:
        ...

    def set_num_columns_per_layer(self, layer_name: str) -> None:
        ...

    def update_num_columns_per_layer(self, layer_name: str, num_columns: int) -> None:
        ...

    def add_training_columns(
        self, layer_type: str, layer_name: str, from_column: Any, to_column: Any
    ) -> None:
        ...


class SelectTrainingDataWidget(iw.VBox):

    name = "Select Training Data"

    def __init__(self, manager: Manager, **kwargs) -> None:
        self._manager = manager

        self.layer_type_dropdown = iw.Dropdown(
            options=["input", "output"],
            value="input",
            description="Choose layer type:",
            style={"description_width": "initial"},
        )
        self.layer_type_dropdown.observe(
            self._on_layer_type_dropdown_value_change, names="value"
        )
        self.layer_dropdown = iw.Dropdown(
            options=self._manager.model.input_names,
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.layer_shape_status = iw.Label(value="Filled layer nodes: None")
        self.columns_label = iw.Label(value="Choose data columns for this layer:")
        self.layer_from_dropdown = iw.Dropdown(
            options=[
                (header, pos) for pos, header in enumerate(self._manager.data.headers)
            ],
            description="From:",
            style={"description_width": "initial"},
        )
        self.layer_from_dropdown.observe(
            self._on_layer_fromto_dropdown_value_change, names="value"
        )
        self.layer_to_dropdown = iw.Dropdown(
            options=[
                (header, pos) for pos, header in enumerate(self._manager.data.headers)
            ],
            description="To:",
            style={"description_width": "initial"},
        )
        self.layer_to_dropdown.observe(
            self._on_layer_fromto_dropdown_value_change, names="value"
        )
        self.selected_columns_num = iw.Label(value="Selected: None")
        self.add_columns_button = iw.Button(description="Add Column(s)")
        self.add_columns_button.on_click(self._on_add_columns_button_clicked)
        self.add_columns_status = iw.Output()

        super().__init__(
            children=[
                self.layer_type_dropdown,
                self.layer_dropdown,
                self.layer_shape_status,
                self.columns_label,
                self.layer_from_dropdown,
                self.layer_to_dropdown,
                self.selected_columns_num,
                self.add_columns_button,
                self.add_columns_status,
            ],
            **kwargs,
        )

    def _update_layer_shape_status(self) -> None:
        self.layer_shape_status.value = f"Filled layer nodes: {self._manager.config.num_columns_per_layer[self.layer_dropdown.value]}/{self._manager.model.layers_shapes[self.layer_dropdown.value]}"

    def _get_selected_columns_num(self) -> Any:
        columns = self._manager.data.headers
        options = list(self.layer_from_dropdown.options)
        value_from = self.layer_from_dropdown.value
        value_to = self.layer_to_dropdown.value
        pair_from = (columns[value_from], value_from)
        pair_to = (columns[value_to], value_to)

        if pair_to not in options:
            return

        return abs(options.index(pair_to) - options.index(pair_from)) + 1

    def _update_selected_columns_num(self) -> None:
        if (
            self.layer_from_dropdown.value is None
            or self.layer_to_dropdown.value is None
        ):
            self.selected_columns_num.value = "Selected: None"
            return

        self.selected_columns_num.value = (
            f"Selected: {self._get_selected_columns_num()} column(s)"
        )

    def _on_layer_type_dropdown_value_change(self, change: Any) -> None:
        if change["new"] == "input":
            self.layer_dropdown.options = self._manager.model.input_names
        else:
            self.layer_dropdown.options = self._manager.model.output_names

    def _on_layer_dropdown_value_change(self, _) -> None:
        self._manager.set_num_columns_per_layer(layer_name=self.layer_dropdown.value)
        self._update_layer_shape_status()

    def _on_layer_fromto_dropdown_value_change(self, _) -> None:
        self._update_selected_columns_num()

    def _update_layer_fromto_dropdown_options(
        self, from_column: Any, to_column: Any
    ) -> None:
        for pair in self.layer_from_dropdown.options:
            if pair[1] in range(from_column, to_column):
                tmp_1 = list(self.layer_from_dropdown.options)
                tmp_2 = list(self.layer_to_dropdown.options)
                tmp_1.remove(pair)
                tmp_2.remove(pair)

                self.layer_from_dropdown.options = tmp_1
                self.layer_to_dropdown.options = tmp_2

    def _on_add_columns_button_clicked(self, _) -> None:
        self.add_columns_status.clear_output(wait=True)

        if self._manager.data.file is None:
            with self.add_columns_status:
                print("Please, upload the file first!\u274C")
            return

        if self._manager.model.instance is None:
            with self.add_columns_status:
                print("Please, upload the model first!\u274C")
            return

        if not self._manager.model.layers:
            with self.add_columns_status:
                print("There are no layers in the model!\u274C")
            return

        if self.layer_dropdown.value is None:
            with self.add_columns_status:
                print("Please, choose the layer first!\u274C")
            return

        if (
            self.layer_from_dropdown.value is None
            or self.layer_to_dropdown.value is None
        ):
            with self.add_columns_status:
                print("Please, choose the data columns first!\u274C")
            return

        current_layer_type = self.layer_type_dropdown.value
        current_layer = self.layer_dropdown.value
        current_counter = self._manager.config.num_columns_per_layer[current_layer]
        current_layer_shape = self._manager.model.layers_shapes[current_layer]
        from_column = self.layer_from_dropdown.value
        to_column = self.layer_to_dropdown.value + 1

        if to_column - 1 < from_column:
            from_column, to_column = to_column - 1, from_column + 1

        new_range = to_column - from_column

        if new_range + current_counter > current_layer_shape:
            with self.add_columns_status:
                print("You've selected more columns than the layer can accept!\u274C")
            return

        self._manager.update_num_columns_per_layer(
            layer_name=current_layer, num_columns=self._get_selected_columns_num()
        )
        self._manager.add_training_columns(
            layer_type=current_layer_type,
            layer_name=current_layer,
            from_column=from_column,
            to_column=to_column,
        )
        self._update_layer_fromto_dropdown_options(
            from_column=from_column, to_column=to_column
        )
        self._update_layer_shape_status()

        with self.add_columns_status:
            print("Your columns have been successfully added!\u2705")

    def _on_widget_state_change(self):
        self.add_columns_status.clear_output(wait=True)

        if self.layer_type_dropdown.value == "input":
            self.layer_dropdown.options = self._manager.model.input_names
        else:
            self.layer_dropdown.options = self._manager.model.output_names

        self.layer_from_dropdown.options = self.layer_to_dropdown.options = [
            (header, pos) for pos, header in enumerate(self._manager.data.headers)
        ]
