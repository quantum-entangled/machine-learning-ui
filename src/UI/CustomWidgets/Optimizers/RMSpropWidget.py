from typing import Any
import ipywidgets as iw


class RMSpropWidget(iw.VBox):

    name = "RMSprop Optimizer"

    def __init__(self, **kwargs):
        self.learning_rate = iw.BoundedFloatText(
            value=0.001,
            min=0,
            max=0.1,
            step=0.005,
            description="Learning rate:",
            style={"description_width": "initial"},
        )
        self.rho = iw.BoundedFloatText(
            value=0.9,
            min=0,
            max=1,
            step=0.005,
            description="Discounting factor:",
            style={"description_width": "initial"},
        )
        self.momentum = iw.BoundedFloatText(
            value=0,
            min=0,
            max=1,
            step=0.005,
            description="Momentum:",
            style={"description_width": "initial"},
        )

        super().__init__(
            children=[self.learning_rate, self.rho, self.momentum], **kwargs
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate.value,
            "rho": self.rho.value,
            "momentum": self.momentum.value,
        }
