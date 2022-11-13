from dataclasses import dataclass, field
from gc import callbacks
from typing import Any


@dataclass
class Config:
    """Config container."""

    num_headers_per_layer: dict[str, int] = field(default_factory=dict)
    input_training_columns: dict[Any, Any] = field(default_factory=dict)
    output_training_columns: dict[Any, Any] = field(default_factory=dict)
    optimizer: Any = None
    losses: dict[Any, Any] = field(default_factory=dict)
    metrics: dict[Any, Any] = field(default_factory=dict)
    callbacks: list[Any] = field(default_factory=list)
