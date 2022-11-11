from typing import Any

import tensorflow as tf

from DataClasses import WidgetWrapper
from UI.CustomWidgets.Optimizers import AdamWidget, RMSpropWidget, SGDWidget

optimizers: dict[str, Any] = {
    "Adam": WidgetWrapper(instance=tf.keras.optimizers.Adam, widget=AdamWidget),
    "SGD": WidgetWrapper(instance=tf.keras.optimizers.SGD, widget=SGDWidget),
    "RMSprop": WidgetWrapper(
        instance=tf.keras.optimizers.RMSprop, widget=RMSpropWidget
    ),
}
