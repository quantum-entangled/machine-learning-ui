import tensorflow as tf

import mlui.types.classes as t

classes: t.ActivationTypes = {
    "Linear": tf.keras.activations.linear,
    "Tanh": tf.keras.activations.tanh,
    "ReLU": tf.keras.activations.relu,
    "Sigmoid": tf.keras.activations.sigmoid,
    "Softmax": tf.keras.activations.softmax,
}
