from typing import Type

import tensorflow as tf

import mlui.widgets.optimizers as wo

classes: dict[str, tf.keras.optimizers.Optimizer] = {
    "Adam": tf.keras.optimizers.Adam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
}

widgets: dict[str, Type[wo.OptimizerWidget]] = {
    "Adam": wo.Adam,
    "RMSprop": wo.RMSprop,
    "SGD": wo.SGD,
}
