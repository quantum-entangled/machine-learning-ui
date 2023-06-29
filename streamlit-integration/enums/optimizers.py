from typing import Type

from streamlit_integration.CustomOptimizers import AdaMod, Apollo
import tensorflow_addons as tfa
import tensorflow as tf
import widgets.optimizers as wo

classes: dict[str, tf.keras.optimizers.Optimizer] = {
    "Adam": tf.keras.optimizers.Adam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
    "AdaMod": AdaMod,
    "Apollo": Apollo,
    "LAMB": tfa.optimizers.LAMB,
    "Lookahead": tfa.optimizers.Lookahead,
    "RAdam": tfa.optimizers.RectifiedAdam,
}

widgets: dict[str, Type[wo.OptimizerWidget]] = {
    "Adam": wo.Adam,
    "RMSprop": wo.RMSprop,
    "SGD": wo.SGD,
    "AdaMod": wo.AdaMod,
    "Apollo": wo.Apollo,
    "LAMB": wo.LAMB,
    "Lookahead": wo.Lookahead,
    "RAdam": wo.RAdam,
}
