from typing import Type

import tensorflow as tf

from ..widgets import callbacks as wc

classes: dict[str, Type[tf.keras.callbacks.Callback]] = {
    "EarlyStopping": tf.keras.callbacks.EarlyStopping,
    "TerminateOnNaN": tf.keras.callbacks.TerminateOnNaN,
}


widgets: dict[str, Type[wc.CallbackWidget]] = {
    "EarlyStopping": wc.EarlyStopping,
    "TerminateOnNaN": wc.TerminateOnNaN,
}
