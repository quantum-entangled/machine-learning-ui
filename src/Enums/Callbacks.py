from typing import Any

import tensorflow as tf

from DataClasses import WidgetWrapper

optimizers: dict[str, Any] = {
    "CSV Logger": WidgetWrapper(instance=tf.keras.callbacks.CSVLogger, widget=None),
    "Early Stopping": WidgetWrapper(
        instance=tf.keras.callbacks.EarlyStopping, widget=...
    ),
    "Model Checkpoint": WidgetWrapper(
        instance=tf.keras.callbacks.ModelCheckpoint, widget=None
    ),
    "TensorBoard": WidgetWrapper(instance=tf.keras.callbacks.TensorBoard, widget=...),
    "Terminate On NaN": WidgetWrapper(
        instance=tf.keras.callbacks.TerminateOnNaN, widget=None
    ),
}
