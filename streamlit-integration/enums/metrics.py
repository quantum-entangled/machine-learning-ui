from typing import Type

import tensorflow as tf
import tensorflow_addons as tfa

classes: dict[str, Type[tf.keras.metrics.Metric]] = {
    "Mean Absolute Error": tf.keras.metrics.MeanAbsoluteError,
    "Mean Absolute Percentage Error": tf.keras.metrics.MeanAbsolutePercentageError,
    "R Square": tfa.metrics.RSquare,
    "Mean Squared Error": tf.keras.metrics.MeanSquaredError,
    "Root Mean Squared Error": tf.keras.metrics.RootMeanSquaredError,
    "Mean Squared Logarithmic Error": tf.keras.metrics.MeanSquaredLogarithmicError,
    "Poisson": tf.keras.metrics.Poisson,
    "LogCosh": tf.keras.metrics.LogCoshError,
}
