from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from UI.CustomWidgets.Layers import DenseLayerWidget, InputLayerWidget


@dataclass
class Layer:
    instance: Any = None
    widget: Any = None


layers = {
    "Input": Layer(instance=tf.keras.layers.Input, widget=InputLayerWidget),
    "Dense": Layer(instance=tf.keras.layers.Dense, widget=DenseLayerWidget),
}
