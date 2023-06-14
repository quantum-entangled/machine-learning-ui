from typing import Any
import ipywidgets as iw
import tensorflow as tf

Optimizers = [tf.keras.optimizers.SGD(), tf.keras.optimizers.Adam(), tf.keras.optimizers.RMSprop(), tf.keras.optimizers.Adamax(), tf.keras.optimizers.Adagrad()]


class LookaheadWidget(iw.VBox):

    name = "Lookahead Optimizer"

    def __init__(self, **kwargs):
        # Widgets
        self.optimizer = iw.Dropdown(
            options=[("SGD", 0), ("Adam", 1), ("RMSprop", 2), ("Adamax", 3), ("Adagrad", 4)],
            value=0,
            disabled=False,
            description="Optimizer:",
            style={"description_width": "initial"},
        )

        super().__init__(children=[self.optimizer])

    @property
    def params(self) -> dict[str, Any]:
        return {
            "optimizer": Optimizers[self.optimizer.value],
        }
