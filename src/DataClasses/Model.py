from dataclasses import dataclass, field
from typing import Any


@dataclass
class Model:
    """Model container."""

    name: str = ""
    instance: Any = None
    layers: dict[str, Any] = field(default_factory=dict)
    input_shapes: dict[str, int] = field(default_factory=dict)
    output_shapes: dict[str, int] = field(default_factory=dict)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    input_model_columns: dict[str, list[str]] = field(default_factory=dict)
    output_model_columns: dict[str, list[str]] = field(default_factory=dict)
    layers_fullness: dict[str, int] = field(default_factory=dict)
    optimizer: Any = None
    losses: dict[Any, Any] = field(default_factory=dict)
    metrics: dict[Any, Any] = field(default_factory=dict)
    callbacks: list[Any] = field(default_factory=list)
    training_history: dict[Any, Any] = field(default_factory=dict)
