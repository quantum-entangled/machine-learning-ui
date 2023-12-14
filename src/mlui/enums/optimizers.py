import tensorflow as tf

import mlui.widgets.optimizers as widget
from mlui.types.classes import OptimizerTypes
from mlui.types.widgets import OptimizerWidgetTypes

classes: OptimizerTypes = {
    "Adam": tf.keras.optimizers.Adam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
}

widgets: OptimizerWidgetTypes = {
    "Adam": widget.Adam,
    "RMSprop": widget.RMSprop,
    "SGD": widget.SGD,
}
