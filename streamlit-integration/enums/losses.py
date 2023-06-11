from typing import Type

import tensorflow as tf

classes: dict[str, Type[tf.keras.losses.Loss]] = {
    "Mean Absolute Error": tf.keras.losses.MeanAbsoluteError,
    "Mean Absolute Percentage Error": tf.keras.losses.MeanAbsolutePercentageError,
    "Mean Squared Error": tf.keras.losses.MeanSquaredError,
    "Mean Squared Logarithmic Error": tf.keras.losses.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.losses.Poisson,
    "LogCosh": tf.keras.losses.LogCosh,
}
