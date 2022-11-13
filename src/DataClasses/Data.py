from dataclasses import dataclass, field
from typing import Any


@dataclass
class Data:
    """Data container."""

    file: Any = None
    headers: list[Any] = field(default_factory=list)
