from typing import Any

import tensorflow as tf

from src.DataClasses import WidgetWrapper
from src.UI.CustomWidgets.Callbacks import (
    CSVLoggerWidget,
    EarlyStoppingWidget,
    ModelCheckpointWidget,
    TensorBoardWidget,
)

callbacks: dict[str, Any] = {
    "CSV Logger": WidgetWrapper(
        instance=tf.keras.callbacks.CSVLogger, widget=CSVLoggerWidget
    ),
    "Early Stopping": WidgetWrapper(
        instance=tf.keras.callbacks.EarlyStopping, widget=EarlyStoppingWidget
    ),
    "Model Checkpoint": WidgetWrapper(
        instance=tf.keras.callbacks.ModelCheckpoint, widget=ModelCheckpointWidget
    ),
    "TensorBoard": WidgetWrapper(
        instance=tf.keras.callbacks.TensorBoard, widget=TensorBoardWidget
    ),
    "Terminate On NaN": WidgetWrapper(
        instance=tf.keras.callbacks.TerminateOnNaN, widget=None
    ),
}
