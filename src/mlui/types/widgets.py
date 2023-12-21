import typing

import mlui.widgets as widgets

# Layers
LayerWidgetType: typing.TypeAlias = typing.Type[widgets.layers.LayerWidget]
LayerWidgetTypes: typing.TypeAlias = dict[str, LayerWidgetType]

# Optimizers
OptimizerWidgetType: typing.TypeAlias = typing.Type[widgets.optimizers.OptimizerWidget]
OptimizerWidgetTypes: typing.TypeAlias = dict[str, OptimizerWidgetType]

# Callbacks
CallbackWidgetType: typing.TypeAlias = typing.Type[widgets.callbacks.CallbackWidget]
CallbackWidgetTypes: typing.TypeAlias = dict[str, CallbackWidgetType]
