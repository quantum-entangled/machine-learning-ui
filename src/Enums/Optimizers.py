import tensorflow as tf

from src.DataClasses import WidgetWrapper
from src.UI.CustomWidgets.Optimizers import AdamWidget, RMSpropWidget, SGDWidget

optimizers: dict[str, WidgetWrapper] = {
    "Adam": WidgetWrapper(instance=tf.keras.optimizers.Adam, widget=AdamWidget),
    "SGD": WidgetWrapper(instance=tf.keras.optimizers.SGD, widget=SGDWidget),
    "RMSprop": WidgetWrapper(
        instance=tf.keras.optimizers.RMSprop, widget=RMSpropWidget
    ),
}
