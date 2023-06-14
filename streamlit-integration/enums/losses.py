from typing import Type

import tensorflow as tf

classes: dict[str, Type[tf.keras.losses.Loss]] = {
    "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError,
    "MeanAbsolutePercentageError": tf.keras.losses.MeanAbsolutePercentageError,
    "MeanSquaredError": tf.keras.losses.MeanSquaredError,
    "MeanSquaredLogarithmicError": tf.keras.losses.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.losses.Poisson,
    "LogCosh": tf.keras.losses.LogCosh,
}
