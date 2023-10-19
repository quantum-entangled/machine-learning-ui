from typing import Any
import ipywidgets as iw
import tensorflow as tf


optimizer = iw.Dropdown(
            #optinons=[("Adam", tf.keras.optimizers.Adam()), ("SGD", tf.keras.optimizers.SGD()), ("RMSprop", tf.keras.optimizers.RMSprop()),
            #         ("Adamax", tf.keras.optimizers.Adamax()), ("Adagrad", tf.keras.optimizers.Adagrad())],
            #options=[("Adam", 1), ("SGD", 2), ("RMSprop", 3), ("Adamax", 4), ("Adagrad", 5)],
            options=[("SGD", 0), ("Adam", 1), ("RMSprop", 2), ("Adamax", 3), ("Adagrad", 4)],
            value=2,
            disabled=False,
            description="Optimizer:",
            style={"description_width": "initial"},
        )

print(type(tf.keras.optimizers.Adam()))
