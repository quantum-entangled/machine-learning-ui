from typing import Any

import ipywidgets as iw


class SelectModelColumnsWidget(iw.VBox):

    name = "Select Model Columns"

    def __init__(self, data_manager: Any, model_manager: Any, **kwargs) -> None:
        self.data_manager = data_manager
        self.model_manager = model_manager

        self.layer_type_dropdown = iw.Dropdown(
            description="Choose layer type:",
            style={"description_width": "initial"},
        )
        self.layer_dropdown = iw.Dropdown(
            options=self.model_manager.model.input_names,
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.layer_shape_status = iw.Label(value="Filled layer nodes: None")
        self.columns_label = iw.Label(value="Choose data columns for this layer:")
        self.from_column_dropdown = iw.Dropdown(
            options=[
                (header, pos)
                for pos, header in enumerate(self.data_manager.data.columns)
            ],
            description="From:",
            style={"description_width": "initial"},
        )
        self.to_column_dropdown = iw.Dropdown(
            options=[
                (header, pos)
                for pos, header in enumerate(self.data_manager.data.columns)
            ],
            description="To:",
            style={"description_width": "initial"},
        )
        self.selected_columns_num = iw.Label(value="Selected: 0 column(s)")
        self.add_columns_button = iw.Button(description="Add Column(s)")
        self.add_columns_status = iw.Output()

        self.layer_type_dropdown.observe(
            self._on_layer_type_dropdown_value_change, names="value"
        )
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.from_column_dropdown.observe(
            self._on_from_column_dropdown_value_change, names="value"
        )
        self.to_column_dropdown.observe(
            self._on_to_column_dropdown_value_change, names="value"
        )
        self.add_columns_button.on_click(self._on_add_columns_button_clicked)

        super().__init__(
            children=[
                self.layer_type_dropdown,
                self.layer_dropdown,
                self.layer_shape_status,
                self.columns_label,
                self.from_column_dropdown,
                self.to_column_dropdown,
                self.selected_columns_num,
                self.add_columns_button,
                self.add_columns_status,
            ],
            **kwargs,
        )

        self.layer = self.layer_dropdown.value

    def _set_layer_shape_status(self) -> None:
        if self.layer in self.model_manager.model.input_shapes:
            shape = self.model_manager.model.input_shapes[self.layer]
        else:
            shape = self.model_manager.model.output_shapes[self.layer]

        self.layer_shape_status.value = f"Filled layer nodes: {self.data_manager.get_num_columns_per_layer(self.layer)}/{shape}"

    def _get_selected_columns_num(self) -> int:
        options_from = list(self.from_column_dropdown.options)
        options_to = list(self.to_column_dropdown.options)

        if not options_from or not options_to:
            return 0

        pair_from = (
            self.data_manager.data.columns[self.from_column_dropdown.value],
            self.from_column_dropdown.value,
        )
        pair_to = (
            self.data_manager.data.columns[self.to_column_dropdown.value],
            self.to_column_dropdown.value,
        )

        return abs(options_to.index(pair_to) - options_from.index(pair_from)) + 1

    def _set_selected_columns_num(self) -> None:
        self.selected_columns_num.value = (
            f"Selected: {self._get_selected_columns_num()} column(s)"
        )

    def _on_layer_type_dropdown_value_change(self, change: Any) -> None:
        if change["new"] == "input":
            self.layer_dropdown.options = self.model_manager.model.input_names
            self.layer_dropdown.value = self.model_manager.model.input_names[0]
        else:
            self.layer_dropdown.options = self.model_manager.model.output_names
            self.layer_dropdown.value = self.model_manager.model.output_names[0]

    def _on_layer_dropdown_value_change(self, change: Any) -> None:
        self.layer = change["new"]
        self._set_layer_shape_status()

    def _on_from_column_dropdown_value_change(self, _) -> None:
        self._set_selected_columns_num()

    def _on_to_column_dropdown_value_change(self, _) -> None:
        self._set_selected_columns_num()

    def _set_from_to_column_options(self, from_index: Any, to_index: Any) -> None:
        for pair in self.from_column_dropdown.options:
            if pair[1] in range(from_index, to_index + 1):
                new_options = list(self.from_column_dropdown.options)
                new_options.remove(pair)

                self.from_column_dropdown.options = new_options
                self.to_column_dropdown.options = new_options

    def _on_add_columns_button_clicked(self, _) -> None:
        self.add_columns_status.clear_output(wait=True)

        if self.data_manager.data.file is None:
            with self.add_columns_status:
                print("Please, upload the file first!\u274C")
            return

        if self.model_manager.model.instance is None:
            with self.add_columns_status:
                print("Please, upload the model first!\u274C")
            return

        if not self.model_manager.model.layers:
            with self.add_columns_status:
                print("There are no layers in the model!\u274C")
            return

        if self.layer_dropdown.value is None:
            with self.add_columns_status:
                print("Please, choose the layer first!\u274C")
            return

        from_index = self.from_column_dropdown.value
        to_index = self.to_column_dropdown.value

        if from_index is None or to_index is None:
            with self.add_columns_status:
                print("Please, choose the data columns first!\u274C")
            return

        if to_index < from_index:
            from_index, to_index = to_index, from_index

        if self.layer in self.model_manager.model.input_shapes:
            shape = self.model_manager.model.input_shapes[self.layer]
        else:
            shape = self.model_manager.model.output_shapes[self.layer]

        columns_range = to_index - from_index + 1

        if (
            columns_range + self.data_manager.get_num_columns_per_layer(self.layer)
            > shape
        ):
            with self.add_columns_status:
                print("You've selected more columns than the layer can accept!\u274C")
            return

        self.data_manager.add_model_columns(
            layer_type=self.layer_type_dropdown.value,
            layer_name=self.layer,
            from_column=from_index,
            to_column=to_index + 1,
        )
        self.data_manager.set_num_columns_per_layer(
            layer_name=self.layer, num_columns=columns_range
        )

        self._set_from_to_column_options(from_index=from_index, to_index=to_index)
        self._set_layer_shape_status()

        with self.add_columns_status:
            print("Your columns have been successfully added!\u2705")

    def _on_widget_state_change(self):
        self.add_columns_status.clear_output()

        if (
            not self.data_manager.data.columns
            or not self.model_manager.model.input_names
            or not self.model_manager.model.output_names
        ):
            return

        if sum(self.data_manager.data.num_columns_per_layer.values()) > 0:
            return

        from_to_options = [
            (header, pos) for pos, header in enumerate(self.data_manager.data.columns)
        ]

        self.layer_type_dropdown.options = ["input", "output"]
        self.layer_type_dropdown.value = "input"

        self.from_column_dropdown.options = from_to_options
        self.from_column_dropdown.value = from_to_options[0][1]

        self.to_column_dropdown.options = from_to_options
        self.to_column_dropdown.value = from_to_options[0][1]
