from typing import Type

import tensorflow as tf
import widgets.layers as wl

classes: dict[str, Type[tf.keras.layers.Layer]] = {
    "Input": tf.keras.Input,
    "Dense": tf.keras.layers.Dense,
    "Concatenate": tf.keras.layers.Concatenate,
}

widgets: dict[str, Type[wl.LayerWidget]] = {
    "Input": wl.Input,
    "Dense": wl.Dense,
    "Concatenate": wl.Concatenate,
}
