from dataclasses import dataclass, field
from typing import Any


@dataclass
class Model:
    """Model container."""

    name: str | None = None
    instance: Any = None
    layers: dict[Any, Any] = field(default_factory=dict)
    input_shapes: dict[Any, Any] = field(default_factory=dict)
    output_shapes: dict[Any, Any] = field(default_factory=dict)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
