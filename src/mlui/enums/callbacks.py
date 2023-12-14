import tensorflow as tf

import mlui.widgets.callbacks as widget
from mlui.types.classes import CallbackTypes
from mlui.types.widgets import CallbackWidgetTypes

classes: CallbackTypes = {
    "EarlyStopping": tf.keras.callbacks.EarlyStopping,
    "TerminateOnNaN": tf.keras.callbacks.TerminateOnNaN,
}

widgets: CallbackWidgetTypes = {
    "EarlyStopping": widget.EarlyStopping,
    "TerminateOnNaN": widget.TerminateOnNaN,
}
