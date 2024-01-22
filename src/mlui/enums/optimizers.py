import tensorflow as tf

import mlui.types.classes as ct
import mlui.types.widgets as wt
import mlui.widgets.optimizers as widget

classes: ct.OptimizerTypes = {
    "Adam": tf.keras.optimizers.Adam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
}

widgets: wt.OptimizerWidgetTypes = {
    "Adam": widget.Adam,
    "RMSprop": widget.RMSprop,
    "SGD": widget.SGD,
}
