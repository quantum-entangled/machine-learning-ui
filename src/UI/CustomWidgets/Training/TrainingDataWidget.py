from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for teaining managers."""

    @property
    def data(self) -> Any:
        ...

    @property
    def model(self) -> Any:
        ...

    def check_instances(self, output_handler: Any) -> bool:
        ...

    def add_training_columns(
        self, layer_type: str, layer_name: str, indices: Any
    ) -> None:
        ...


class TrainingDataWidget(iw.VBox):

    name = "Choose Training Data"

    def __init__(self, manager: Manager, **kwargs) -> None:
        self._manager = manager

        self.select_data_button = iw.Button(description="Select Training Data")
        self.select_data_button.on_click(self._on_select_data_button_clicked)
        self.select_data_output = iw.Output()

        super().__init__(
            children=[self.select_data_button, self.select_data_output], **kwargs
        )

    def _on_select_data_button_clicked(self, _):
        status = self._manager.check_instances(output_handler=self.select_data_output)

        if not status:
            return

        self.column_names = self._manager.data.headers
        self.column_indices = list(range(len(self.column_names)))
        self.input_layer_names = [
            layer[0]
            for layer in self._manager.model.instance.get_config()["input_layers"]
        ]
        self.output_layer_names = [
            layer[0]
            for layer in self._manager.model.instance.get_config()["output_layers"]
        ]
        self.layer_column_counters = {
            name: 0 for name in self.input_layer_names + self.output_layer_names
        }
        self.input_layer_shapes = {
            name: self._manager.model.instance.get_layer(name).input_shape[0][1]
            for name in self.input_layer_names
        }
        self.output_layer_shapes = {
            name: self._manager.model.instance.get_layer(name).output_shape[1]
            for name in self.output_layer_names
        }
        self.layer_shapes = self.input_layer_shapes | self.output_layer_shapes

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
            options=self.input_layer_names,
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.layer_shape_status = iw.Label(
            value=f"Filled layer nodes: {self.layer_column_counters[self.layer_dropdown.value]}/{self.layer_shapes[self.layer_dropdown.value]}"
        )
        self.columns_label = iw.Label(value="Choose data columns for this layer:")
        self.layer_from_dropdown = iw.Dropdown(
            options=list(
                zip(
                    [self.column_names[i] for i in self.column_indices],
                    self.column_indices,
                )
            ),
            description="From:",
            style={"description_width": "initial"},
        )
        self.layer_from_dropdown.observe(
            self._on_layer_from_dropdown_value_change, names="value"
        )
        self.layer_to_dropdown = iw.Dropdown(
            options=list(
                zip(
                    [self.column_names[i] for i in self.column_indices],
                    self.column_indices,
                )
            ),
            description="To:",
            style={"description_width": "initial"},
        )
        self.layer_to_dropdown.observe(
            self._on_layer_to_dropdown_value_change, names="value"
        )
        self.selected_columns_num = iw.Label(
            value=f"Selected: {abs(self.layer_to_dropdown.value - self.layer_from_dropdown.value) + 1} column(s)"
        )
        self.add_columns_button = iw.Button(description="Add Column(s)")
        self.add_columns_button.on_click(self._on_add_columns_button_clicked)
        self.add_columns_status = iw.Output()

        self.children = [
            self.layer_type_dropdown,
            self.layer_dropdown,
            self.layer_shape_status,
            self.columns_label,
            self.layer_from_dropdown,
            self.layer_to_dropdown,
            self.selected_columns_num,
            self.add_columns_button,
            self.add_columns_status,
        ]

    def _on_layer_type_dropdown_value_change(self, change: Any) -> None:
        if change["new"] == "input":
            self.layer_dropdown.options = self.input_layer_names
        else:
            self.layer_dropdown.options = self.output_layer_names

        self.layer_shape_status.value = f"Filled layer nodes: {self.layer_column_counters[self.layer_dropdown.value]}/{self.layer_shapes[self.layer_dropdown.value]}"

    def _on_layer_dropdown_value_change(self, change: Any) -> None:
        self.layer_shape_status.value = f"Filled layer nodes: {self.layer_column_counters[change['new']]}/{self.layer_shapes[change['new']]}"

    def _on_layer_from_dropdown_value_change(self, change: Any) -> None:
        if not change["new"]:
            self.selected_columns_num.value = "Selected: 0 column(s)"
            return

        self.selected_columns_num.value = f"Selected: {abs(self.layer_to_dropdown.value - change['new']) + 1} column(s)"

    def _on_layer_to_dropdown_value_change(self, change: Any) -> None:
        if not change["new"]:
            self.selected_columns_num.value = "Selected: 0 column(s)"
            return

        self.selected_columns_num.value = f"Selected: {abs(change['new'] - self.layer_from_dropdown.value) + 1} column(s)"

    def _on_add_columns_button_clicked(self, _) -> None:
        self.add_columns_status.clear_output(wait=True)

        current_layer_type = self.layer_type_dropdown.value
        current_layer = self.layer_dropdown.value
        current_counter = self.layer_column_counters[current_layer]
        current_layer_shape = self.layer_shapes[current_layer]
        from_column = self.layer_from_dropdown.value
        to_column = self.layer_to_dropdown.value + 1

        if to_column - 1 < from_column:
            from_column, to_column = to_column - 1, from_column + 1

        new_range = to_column - from_column

        if new_range + current_counter > current_layer_shape:
            with self.add_columns_status:
                print("You've selected more columns than the layer can accept!\u274C")
            return

        self.column_indices = sorted(
            set(self.column_indices) - set(range(from_column, to_column))
        )
        self._manager.add_training_columns(
            layer_type=current_layer_type,
            layer_name=current_layer,
            indices=list(range(from_column, to_column)),
        )
        self.layer_from_dropdown.options = self.layer_to_dropdown.options = list(
            zip(
                [self.column_names[i] for i in self.column_indices],
                self.column_indices,
            )
        )
        self.layer_column_counters[current_layer] = current_counter + new_range
        self.layer_shape_status.value = f"Filled layer nodes: {self.layer_column_counters[current_layer]}/{self.layer_shapes[current_layer]}"

        with self.add_columns_status:
            print("Your columns have been successfully added!\u2705")
