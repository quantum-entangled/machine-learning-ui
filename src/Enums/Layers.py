from typing import Any

import tensorflow as tf

from src.DataClasses import WidgetWrapper
from src.UI.CustomWidgets.Layers import (
    ConcatenateLayerWidget,
    DenseLayerWidget,
    InputLayerWidget,
)

layers: dict[str, Any] = {
    "Input": WidgetWrapper(instance=tf.keras.Input, widget=InputLayerWidget),
    "Dense": WidgetWrapper(instance=tf.keras.layers.Dense, widget=DenseLayerWidget),
    "Concatenate": WidgetWrapper(
        instance=tf.keras.layers.Concatenate, widget=ConcatenateLayerWidget
    ),
}
