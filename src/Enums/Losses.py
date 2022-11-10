from typing import Any

import tensorflow as tf

losses: dict[Any, Any] = {
    "Mean Absolute Error": tf.keras.losses.MeanAbsoluteError,
    "Mean Absolute Percentage Error": tf.keras.losses.MeanAbsolutePercentageError,
    "Mean Squared Error": tf.keras.losses.MeanSquaredError,
    "Mean Squared Logarithmic Error": tf.keras.losses.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.losses.Poisson,
    "LogCosh": tf.keras.losses.LogCosh,
}
