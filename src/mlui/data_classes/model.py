from dataclasses import dataclass, field
from typing import Any, TypeAlias

import pandas as pd
import tensorflow as tf

Metrics: TypeAlias = dict[str, tf.keras.metrics.Metric]
LayerMetrics: TypeAlias = dict[str, Metrics]
Callbacks: TypeAlias = list[tf.keras.callbacks.Callback]


@dataclass
class Model:
    """Model container."""

    name: str = ""
    instance: Any = None
    input_layers: dict[str, Any] = field(default_factory=dict)
    output_layers: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, Any] = field(default_factory=dict)
    input_shapes: dict[str, int] = field(default_factory=dict)
    output_shapes: dict[str, int] = field(default_factory=dict)
    compiled: bool = False
    optimizer: Any = None
    losses: dict[str, Any] = field(default_factory=dict)
    metrics: LayerMetrics = field(default_factory=dict)
    callbacks: Callbacks = field(default_factory=list)
    training_history: pd.DataFrame = field(default_factory=pd.DataFrame)
