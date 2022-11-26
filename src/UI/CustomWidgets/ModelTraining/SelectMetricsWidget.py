from typing import Any, Protocol

import ipywidgets as iw

from Enums.Metrics import metrics


class ModelManager(Protocol):
    """Protocol for training managers."""

    @property
    def model(self) -> Any:
        ...

    def add_metric(self, layer_name: str, metric: Any) -> None:
        ...


class SelectMetricsWidget(iw.VBox):

    name = "Select Model Metrics"

    def __init__(self, model_manager: ModelManager, **kwargs):
        self.model_manager = model_manager

        self.layer_dropdown = iw.Dropdown(
            options=list(self.model_manager.model.output_names),
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.metric_dropdown = iw.Dropdown(
            options=list(metrics),
            description="Choose loss function:",
            style={"description_width": "initial"},
        )
        self.add_metric_button = iw.Button(description="Add Loss Function")
        self.add_metric_button.on_click(self._on_add_metric_button_clicked)
        self.metric_status = iw.Output()

        super().__init__(
            children=[
                self.layer_dropdown,
                self.metric_dropdown,
                self.add_metric_button,
                self.metric_status,
            ],
            **kwargs,
        )

    def _on_add_metric_button_clicked(self, _) -> None:
        self.metric_status.clear_output(wait=True)

        if not self.model_manager.model.instance:
            with self.metric_status:
                print("Please, upload the model first!\u274C")
            return

        if not self.layer_dropdown.value:
            with self.metric_status:
                print("There are no output layers in the model!\u274C")
            return

        self.model_manager.add_metric(
            layer_name=self.layer_dropdown.value,
            metric=metrics[self.metric_dropdown.value](),
        )

        with self.metric_status:
            print(f"Metric has been successfully added!\u2705")

    def _on_widget_state_change(self) -> None:
        self.metric_status.clear_output(wait=True)

        options = self.model_manager.model.output_names

        if options:
            self.layer_dropdown.options = options
            self.layer_dropdown.value = options[0]
