import tensorflow as tf

import mlui.types.classes as ct
import mlui.types.widgets as wt
import mlui.widgets.layers as widget

classes: ct.LayerTypes = {
    "Input": tf.keras.Input,
    "Dense": tf.keras.layers.Dense,
    "Concatenate": tf.keras.layers.Concatenate,
    "BatchNormalization": tf.keras.layers.BatchNormalization,
    "Dropout": tf.keras.layers.Dropout,
}

widgets: wt.LayerWidgetTypes = {
    "Input": widget.Input,
    "Dense": widget.Dense,
    "Concatenate": widget.Concatenate,
    "BatchNormalization": widget.BatchNormalization,
    "Dropout": widget.Dropout,
}
