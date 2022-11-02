import tensorflow as tf
from DataClasses import Layer
from UI.CustomWidgets.Layers import (
    ConcatenateLayerWidget,
    DenseLayerWidget,
    InputLayerWidget,
)

layers = {
    "Input": Layer(instance=tf.keras.layers.Input, widget=InputLayerWidget),
    "Dense": Layer(instance=tf.keras.layers.Dense, widget=DenseLayerWidget),
    "Concatenate": Layer(
        instance=tf.keras.layers.Concatenate, widget=ConcatenateLayerWidget
    ),
}
