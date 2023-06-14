from typing import Any
import ipywidgets as iw


class RAdamWidget(iw.VBox):

    name = "RAdam Optimizer"

    def __init__(self, **kwargs):
        # Widgets
        self.learning_rate = iw.BoundedFloatText(
            value=0.001,
            min=0,
            max=0.1,
            step=0.005,
            description="Learning rate:",
            style={"description_width": "initial"},
        )
        self.beta_1 = iw.BoundedFloatText(
            value=0.9,
            min=0,
            max=1,
            step=0.005,
            description="Decay 1:",
            style={"description_width": "initial"},
        )
        self.beta_2 = iw.BoundedFloatText(
            value=0.999,
            min=0,
            max=1,
            step=0.005,
            description="Decay 2:",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.learning_rate, self.beta_1, self.beta_2])

    @property
    def params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate.value,
            "beta_1": self.beta_1.value,
            "beta_2": self.beta_2.value,
        }
