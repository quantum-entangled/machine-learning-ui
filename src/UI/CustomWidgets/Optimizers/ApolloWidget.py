from typing import Any
import ipywidgets as iw
import tensorflow as tf

WD_types = ["L2", "decoupled", "stable"]


class ApolloWidget(iw.VBox):

    name = "Apollo Optimizer"

    def __init__(self, **kwargs):
        # Widgets
        self.learning_rate = iw.BoundedFloatText(
            value=0.01,
            min=0,
            max=0.1,
            step=0.005,
            description="Learning rate:",
            style={"description_width": "initial"},
        )
        self.beta = iw.BoundedFloatText(
            value=0.9,
            min=0,
            max=1,
            step=0.005,
            description="Beta:",
            style={"description_width": "initial"},
        )
        self.weight_decay = iw.BoundedFloatText(
            value=0,
            min=0,
            max=0.001,
            step=0.00005,
            description="Weight decay:",
            style={"description_width": "initial"},
        )
        self.weight_decay_type = iw.Dropdown(
            options=[("L2", 0), ("Decoupled", 1), ("Stable", 2)],
            value=0,
            disabled=False,
            description="Weight decay type:",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.learning_rate, self.beta, self.weight_decay, self.weight_decay_type])

    @property
    def params(self) -> dict[str, Any]:
        return {
            "learning_rate": tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.learning_rate.value,
                first_decay_steps=1000,
            ),
            "beta": self.beta.value,
            "weight_decay": self.weight_decay.value,
            "weight_decay_type": WD_types[self.weight_decay_type.value],
        }
