import tensorflow as tf
import tensorflow_addons as tfa

from src.CustomOptimizers import AdaMod
from src.CustomOptimizers import Apollo

from src.DataClasses import WidgetWrapper
from src.UI.CustomWidgets.Optimizers import AdamWidget, RMSpropWidget, SGDWidget, AdaModWidget, \
    ApolloWidget, LAMBWidget, LookaheadWidget, RAdamWidget

optimizers: dict[str, WidgetWrapper] = {
    "Adam": WidgetWrapper(instance=tf.keras.optimizers.Adam, widget=AdamWidget),
    "SGD": WidgetWrapper(instance=tf.keras.optimizers.SGD, widget=SGDWidget),
    "RMSprop": WidgetWrapper(
        instance=tf.keras.optimizers.RMSprop, widget=RMSpropWidget
    ),
    "AdaMod": WidgetWrapper(instance=AdaMod, widget=AdaModWidget),
    "Apollo": WidgetWrapper(instance=Apollo, widget=ApolloWidget),
    "LAMB": WidgetWrapper(instance=tfa.optimizers.LAMB, widget=LAMBWidget),
    "Lookahead": WidgetWrapper(instance=tfa.optimizers.Lookahead, widget=LookaheadWidget),
    "RAdam": WidgetWrapper(instance=tfa.optimizers.RectifiedAdam, widget=RAdamWidget),
}
