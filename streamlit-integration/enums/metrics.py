from typing import Type

import tensorflow as tf
import tensorflow_addons as tfa

classes: dict[str, Type[tf.keras.metrics.Metric]] = {
    "MeanAbsoluteError": tf.keras.metrics.MeanAbsoluteError,
    "MeanAbsolutePercentageError": tf.keras.metrics.MeanAbsolutePercentageError,
    "RSquare": tfa.metrics.RSquare,
    "MeanSquaredError": tf.keras.metrics.MeanSquaredError,
    "RootMeanSquaredError": tf.keras.metrics.RootMeanSquaredError,
    "MeanSquaredLogarithmicError": tf.keras.metrics.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.metrics.Poisson,
    "LogCosh": tf.keras.metrics.LogCoshError,
}
