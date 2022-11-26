from dataclasses import dataclass, field
from typing import Any


@dataclass
class Data:
    """Data container."""

    file: Any = None
    columns: list[Any] = field(default_factory=list)
    num_columns_per_layer: dict[str, int] = field(default_factory=dict)
    input_training_columns: dict[Any, Any] = field(default_factory=dict)
    output_training_columns: dict[Any, Any] = field(default_factory=dict)
