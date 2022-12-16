from typing import Any, Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error
from src.Enums.Metrics import metrics
from src.Enums.SuccessMessages import Success


class ModelManager(Protocol):
    """Protocol for training managers."""

    def model_exists(self) -> bool:
        ...

    def add_metric(self, layer: str, metric: Any) -> None:
        ...

    @property
    def metrics(self) -> dict[str, list[Any]]:
        ...


class SelectMetricsWidget(iw.VBox):

    name = "Select Model Metrics"

    def __init__(self, model_manager: ModelManager, **kwargs):
        # Managers
        self.model_manager = model_manager

        # Widgets
        self.layer_dropdown = iw.Dropdown(
            description="Choose layer:",
            style={"description_width": "initial"},
        )
        self.metrics_dropdown = iw.Dropdown(
            options=list(metrics),
            description="Select metrics:",
            style={"description_width": "initial"},
        )
        self.add_metric_button = iw.Button(description="Add Loss Function")
        self.metric_status = iw.Output()

        # Callbacks
        self.layer_dropdown.observe(self._on_layer_dropdown_value_change, names="value")
        self.add_metric_button.on_click(self._on_add_metric_button_clicked)

        super().__init__(
            children=[
                self.layer_dropdown,
                self.metrics_dropdown,
                self.add_metric_button,
                self.metric_status,
            ]
        )

    def _on_layer_dropdown_value_change(self, _) -> None:
        self.metric_status.clear_output()

    def _on_add_metric_button_clicked(self, _) -> None:
        self.metric_status.clear_output(wait=True)

        with self.metric_status:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            layer = self.layer_dropdown.value
            metric = metrics[self.metrics_dropdown.value]

            if not layer:
                print(Error.NO_OUTPUT_LAYERS)
                return

            if metric in [type(metric) for metric in self.model_manager.metrics[layer]]:
                print(Error.SAME_METRIC)
                return

            self.model_manager.add_metric(layer=layer, metric=metric)

            print(Success.LOSS_ADDED)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.metric_status.clear_output()

        options = list(self.model_manager.metrics)

        self.layer_dropdown.options = options
        self.layer_dropdown.value = options[0] if options else None
