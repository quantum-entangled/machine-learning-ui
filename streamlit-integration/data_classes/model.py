from dataclasses import dataclass, field
from typing import Any


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
    metrics: dict[str, list[Any]] = field(default_factory=dict)
    callbacks: list[Any] = field(default_factory=list)
    training_history: dict[str, list[Any]] = field(default_factory=dict)
