import tensorflow as tf

import mlui.widgets.layers as widget
from mlui.types.classes import LayerTypes
from mlui.types.widgets import LayerWidgetTypes

classes: LayerTypes = {
    "Input": tf.keras.Input,
    "Dense": tf.keras.layers.Dense,
    "Concatenate": tf.keras.layers.Concatenate,
    "BatchNormalization": tf.keras.layers.BatchNormalization,
    "Dropout": tf.keras.layers.Dropout,
}

widgets: LayerWidgetTypes = {
    "Input": widget.Input,
    "Dense": widget.Dense,
    "Concatenate": widget.Concatenate,
    "BatchNormalization": widget.BatchNormalization,
    "Dropout": widget.Dropout,
}
