import tensorflow as tf

import mlui.types.classes as ct
import mlui.types.widgets as wt
import mlui.widgets.callbacks as widget

classes: ct.CallbackTypes = {
    "EarlyStopping": tf.keras.callbacks.EarlyStopping,
    "TerminateOnNaN": tf.keras.callbacks.TerminateOnNaN,
}

widgets: wt.CallbackWidgetTypes = {
    "EarlyStopping": widget.EarlyStopping,
    "TerminateOnNaN": widget.TerminateOnNaN,
}
