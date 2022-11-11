from typing import Any
import ipywidgets as iw


class EarlyStoppingWidget(iw.VBox):

    name = "EarlyStopping Callback"

    def __init__(self, **kwargs):
        self.min_delta = iw.BoundedFloatText(
            value=0,
            min=0,
            max=10,
            step=0.1,
            description="Min delta:",
            style={"description_width": "initial"},
        )
        self.patience = iw.BoundedFloatText(
            value=10,
            min=0,
            max=50,
            step=1,
            description="Patience:",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.min_delta, self.patience], **kwargs)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "min_delta": self.min_delta.value,
            "patience": self.patience.value,
        }
