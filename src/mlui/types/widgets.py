from typing import Type, TypeAlias

import mlui.widgets as widgets

# Layers
LayerWidgetType: TypeAlias = Type[widgets.layers.LayerWidget]
LayerWidgetTypes: TypeAlias = dict[str, LayerWidgetType]

# Optimizers
OptimizerWidgetType: TypeAlias = Type[widgets.optimizers.OptimizerWidget]
OptimizerWidgetTypes: TypeAlias = dict[str, OptimizerWidgetType]

# Callbacks
CallbackWidgetType: TypeAlias = Type[widgets.callbacks.CallbackWidget]
CallbackWidgetTypes: TypeAlias = dict[str, CallbackWidgetType]
