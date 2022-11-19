from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for training managers."""

    @property
    def config(self) -> Any:
        ...

    def plot_history(self, y: Any, color: Any, same_figure: bool) -> None:
        ...


class PlotHistoryWidget(iw.VBox):

    name = "Plot Training History"

    def __init__(self, manager: Manager, **kwargs):
        self._manager = manager

        self.history_type_dropdown = iw.Dropdown(
            options=list(self._manager.config.training_history),
            description="Choose history type:",
            style={"description_width": "initial"},
        )
        self.color_picker = iw.ColorPicker(
            description="Pick a line color:",
            value="blue",
            style={"description_width": "initial"},
        )
        self.same_figure = iw.Checkbox(
            value=False,
            description="Plot on same figure",
            style={"description_width": "initial"},
        )
        self.plot_button = iw.Button(description="Plot History")
        self.plot_button.on_click(self._on_plot_button_clicked)
        self.plot_output = iw.Output()

        super().__init__(
            children=[
                self.history_type_dropdown,
                self.color_picker,
                self.same_figure,
                self.plot_button,
                self.plot_output,
            ],
            **kwargs
        )

    def _on_plot_button_clicked(self, _) -> None:
        self.plot_output.clear_output(wait=True)

        if not self._manager.config.training_history:
            with self.plot_output:
                print("Please, train the model first!\u274C")
            return

        history_type = self.history_type_dropdown.value
        line_color = self.color_picker.value
        same_figure = self.same_figure.value

        if history_type is None:
            with self.plot_output:
                print("Please, choose the history type first!\u274C")
            return

        with self.plot_output:
            self._manager.plot_history(
                y=history_type, color=line_color, same_figure=same_figure
            )

    def _on_widget_state_change(self):
        self.plot_output.clear_output(wait=True)

        self.history_type_dropdown.options = list(self._manager.config.training_history)
