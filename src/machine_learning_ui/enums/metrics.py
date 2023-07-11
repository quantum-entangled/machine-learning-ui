from typing import Type

import tensorflow as tf

classes: dict[str, Type[tf.keras.metrics.Metric]] = {
    "MeanAbsoluteError": tf.keras.metrics.MeanAbsoluteError,
    "MeanAbsolutePercentageError": tf.keras.metrics.MeanAbsolutePercentageError,
    "MeanSquaredError": tf.keras.metrics.MeanSquaredError,
    "RootMeanSquaredError": tf.keras.metrics.RootMeanSquaredError,
    "MeanSquaredLogarithmicError": tf.keras.metrics.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.metrics.Poisson,
    "LogCoshError": tf.keras.metrics.LogCoshError,
}
