from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """Config container."""

    input_training_indices: dict[Any, Any] = field(default_factory=dict)
    output_training_indices: dict[Any, Any] = field(default_factory=dict)
    optimizer: Any = None
    losses: dict[Any, Any] = field(default_factory=dict)
    metrics: dict[Any, Any] = field(default_factory=dict)
